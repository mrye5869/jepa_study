"""
LeWM 的底层模块：预测器、正则器、Transformer 组件、MLP 等。
这些是从零实现的，不依赖外部库（除 PyTorch 和 einops）。
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════════
# AdaLN-zero 调制函数
# ═══════════════════════════════════════════════════════════════════

def modulate(x, shift, scale):
    """
    Adaptive Layer Normalization 调制。
    对归一化后的特征做：x = x * (1 + scale) + shift
    scale 初始为 0 → 训练初期等价于恒等，模型逐步学会利用条件信号
    """
    return x * (1 + scale) + shift


# ═══════════════════════════════════════════════════════════════════
# SIGReg: 防止表示坍缩的核心正则器
# ═══════════════════════════════════════════════════════════════════

class SIGReg(torch.nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer（单 GPU 实现）。

    核心思想：强制嵌入分布接近标准高斯分布 N(0, I)。
    如果所有嵌入都被高斯分布约束住，它们就不可能坍缩到同一个点上。

    原理 —— Epps-Pulley 检验：
    - 标准高斯分布的特征函数是 φ(t) = exp(-t²/2)
    - 把嵌入投影到 1024 个随机方向上
    - 对每个方向，检查经验特征函数 E[cos(t·proj)] 是否 ≈ φ(t)
    - 在 17 个 t 值（knot）上计算偏差，加权求和
    - 所有方向的偏差平均 → SIGReg loss
    """

    def __init__(self, knots=17, num_proj=1024):
        """
        knots:    检验点的个数（在 [0, 3] 区间均匀分布）
        num_proj: 随机投影方向的数量
        """
        super().__init__()
        self.num_proj = num_proj
        # t ∈ [0, 3]，均匀取 17 个点
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        # 复合梯形积分权重
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        # φ(t) = exp(-t²/2)，标准高斯分布的特征函数
        window = torch.exp(-t.square() / 2.0)
        # 注册为 buffer（不参与梯度，但随模型保存和迁移）
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D) — 嵌入序列（时间 × 批次 × 维度）

        计算流程：
        1. 随机生成 1024 个正交投影方向
        2. 把嵌入投影到这些方向上
        3. 对每个方向计算 Epps-Pulley 统计量
        4. 平均所有方向的统计量 → loss
        """
        # 生成随机投影矩阵，并归一化每列
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # 投影： (T, B, D) @ (D, num_proj) → (T, B, num_proj)
        # 对每个方向、每个 t 值计算 cos(t * proj)
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (T, B, num_proj, knots)

        # Epps-Pulley 统计量：经验特征函数与理论特征函数的偏差
        # E[cos(tX)] 应当 ≈ exp(-t²/2)，E[sin(tX)] 应当 ≈ 0
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # 加权积分 + 按嵌入数量归一化
        statistic = (err @ self.weights) * proj.size(-2)
        # 平均所有投影方向和所有时间步
        return statistic.mean()


# ═══════════════════════════════════════════════════════════════════
# Transformer 基础组件
# ═══════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """标准 Transformer 前馈网络: LayerNorm → Linear → GELU → Dropout → Linear → Dropout"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    缩放点积注意力，支持因果掩码（预测器用它实现自回归，只看过去）。
    使用 PyTorch 的 F.scaled_dot_product_attention（自动启用 FlashAttention）。
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x: (B, T, D)
        causal=True → 因果掩码，t 时刻只能 attend t 及之前的位置
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 拆成 q, k, v
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        # PyTorch 内置的 scaled_dot_product_attention（底层用 FlashAttention 加速）
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


# ═══════════════════════════════════════════════════════════════════
# Transformer Block 变体
# ═══════════════════════════════════════════════════════════════════

class ConditionalBlock(nn.Module):
    """
    条件 Transformer 块（AdaLN-zero 风格）。

    和标准 Block 的关键区别：动作不是拼在输入向量里的，
    而是通过 AdaLN 的 scale/shift 参数"调制"每一层的行为。
    gate 初始化为 0 → 训练初期条件分支被关闭，模型逐步学会使用动作信号。
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        # elementwise_affine=False：LN 不做自己的学习缩放，完全由 AdaLN 控制
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        # 从条件向量 c 生成 6 组调制参数（注意力 3 组 + MLP 3 组）
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        # 关键：最后一层权重和偏置初始化为 0 → gate=0 → 训练初条件信号不起作用
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        x: (B, T, D) — 嵌入序列
        c: (B, T, D) — 条件向量（动作嵌入）
        """
        # 从条件向量生成 6 组参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # 注意：gate * sublayer(x) 而非 x + sublayer(x)
        # gate 初始为 0 → 等价于恒等，模型逐步学会使用注意力和 MLP
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """标准 Transformer 块（无条件，用于普通编码器）"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """
    通用 Transformer，支持普通块和条件块。
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,  # 默认用普通块；预测器用 ConditionalBlock
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            # 条件块需要传入 c，普通块不需要
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# Embedder: 动作编码器
# ═══════════════════════════════════════════════════════════════════

class Embedder(nn.Module):
    """
    将动作向量编码为与嵌入空间维度一致的表示。
    Conv1d(1×1) 做通道混合 + MLP 做语义提升。
    """

    def __init__(
        self,
        input_dim=10,       # 原始动作维度（如 2 轴速度指令）
        smoothed_dim=10,    # 1×1 卷积后的中间维度
        emb_dim=10,         # 最终嵌入维度（应与 embed_dim 一致）
        mlp_scale=4,        # MLP 隐藏层倍数
    ):
        super().__init__()
        # 1×1 卷积做逐时间步的通道混合（不跨时间步，保持时序独立）
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, act_dim) → 输出: (B, T, emb_dim)
        注意 Conv1d 期望 (B, C, L) 格式，所以要先 permute
        """
        x = x.float()
        x = x.permute(0, 2, 1)    # (B, T, D) → (B, D, T)
        x = self.patch_embed(x)    # 1×1 卷积逐时间步融合动作通道
        x = x.permute(0, 2, 1)    # (B, D, T) → (B, T, D)
        x = self.embed(x)          # MLP 提升到嵌入维度
        return x


# ═══════════════════════════════════════════════════════════════════
# MLP: 投影头
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    简单的两层 MLP，用作 projector 和 pred_proj。
    将编码器/预测器的输出映射到共享的嵌入空间，接在两者之间做"对齐"。

    结构: Linear → BatchNorm/LayerNorm → GELU → Linear
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """x: (B*T, D) → (B*T, D)"""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# ARPredictor: 自回归条件预测器
# ═══════════════════════════════════════════════════════════════════

class ARPredictor(nn.Module):
    """
    自回归预测器 —— JEPA 的"大脑"。

    本质是一个因果 Transformer，接收历史嵌入和动作，
    自回归地预测下一步的嵌入。

    关键设计:
    - 内部使用 ConditionalBlock（AdaLN-zero），动作信号通过调制注入
    - 因果注意力掩码确保 t 时刻只看 ≤t 的信息
    - 可学习的位置编码帮助理解时序顺序
    """

    def __init__(
        self,
        *,
        num_frames,        # 最大帧数（等于 history_size）
        depth,             # Transformer 层数
        heads,             # 注意力头数
        mlp_dim,           # FFN 隐藏维度
        input_dim,         # 输入维度（= embed_dim）
        hidden_dim,        # Transformer 内部维度
        output_dim=None,   # 输出维度（默认 = input_dim）
        dim_head=64,       # 每个头的维度
        dropout=0.0,       # 注意力 dropout
        emb_dropout=0.0,   # 嵌入 dropout
    ):
        super().__init__()
        # 可学习的位置编码: (1, num_frames, input_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        # 内部是条件 Transformer（block_class=ConditionalBlock）
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,  # ← 关键：使用条件块
        )

    def forward(self, x, c):
        """
        x: (B, T, d) — 历史帧嵌入
        c: (B, T, act_dim) — 动作嵌入（作为条件注入每个 ConditionalBlock）

        流程:
        1. 加上可学习位置编码
        2. 输入条件 Transformer，每层都接收动作信号作为调制参数
        3. Transformer 内部因果掩码 → t 时刻只能看到 ≤t 的信息
        4. 输出: 每个位置预测的"下一帧嵌入"
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]  # 注入位置信息
        x = self.dropout(x)
        x = self.transformer(x, c)  # 条件 Transformer 前向
        return x
