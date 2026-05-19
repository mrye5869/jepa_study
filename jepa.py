"""JEPA Implementation — 联合嵌入预测架构"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    """断开梯度并深拷贝，用于 rollout 时保存初始状态的快照"""
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):
    """
    JEPA 核心类：编码图像→在嵌入空间里预测未来→用于 MPC 规划。

    组件分工：
    - encoder (ViT):        图像 → 嵌入向量
    - predictor (ARPredictor): 历史嵌入 + 动作 → 未来的嵌入
    - action_encoder:       动作 → 动作嵌入
    - projector:            编码器输出 → 共享嵌入空间（MLP）
    - pred_proj:            预测器输出 → 共享嵌入空间（MLP）
    """

    def __init__(
        self,
        encoder,          # ViT，逐帧编码图像
        predictor,        # ARPredictor，条件自回归 Transformer
        action_encoder,   # Embedder，动作编码器
        projector=None,   # 编码器侧的投影 MLP
        pred_proj=None,   # 预测器侧的投影 MLP
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    # ═══════════════════════════════════════════════
    # 训练和推理共用
    # ═══════════════════════════════════════════════

    def encode(self, info):
        """
        将图像和动作编码为嵌入向量。

        输入: info['pixels'] (B, T, C, H, W)  — T帧图像
              info['action'] (B, T, act_dim)   — T个动作（可选）
        输出: info['emb']     (B, T, 192)       — T帧对应的嵌入
              info['act_emb'] (B, T, 192)       — T个动作对应的嵌入
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        # 关键：把时间维拍平到 batch 维，(B,T,C,H,W) → (B*T,C,H,W)
        # 这样 ViT 把每一帧当作独立图像编码，不做时序融合
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        # 取 ViT 的 [CLS] token 作为整张图的表示向量
        pixels_emb = output.last_hidden_state[:, 0]
        # 通过 projector MLP 映射到共享嵌入空间
        emb = self.projector(pixels_emb)
        # 恢复时间维：(B*T, D) → (B, T, D)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """
        根据历史嵌入和动作，预测下一帧的嵌入。

        输入: emb     (B, T, D) — 历史帧的嵌入序列
              act_emb (B, T, A_emb) — 对应的动作嵌入
        输出: preds   (B, T, D) — 每步预测的下一帧嵌入
        """
        # ARPredictor 内部是因果 Transformer，t 时刻只能看 ≤t 的信息
        preds = self.predictor(emb, act_emb)
        # 通过 pred_proj MLP 映射到共享嵌入空间
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    # ═══════════════════════════════════════════════
    # 以下仅用于推理（评估 / MPC 规划）
    # ═══════════════════════════════════════════════

    def rollout(self, info, action_sequence, history_size: int = 3):
        """
        自回归地推演未来状态嵌入。不碰任何新图像，纯在嵌入空间里滚。

        输入:
            action_sequence: (B, S, T, action_dim)
                S — 候选动作序列的数量（如 64 条）
                T — 每条序列的时间跨度
            history_size — 预测时往回看几帧（默认 3）

        流程:
            1. 编码初始图像 → 得到初始嵌入
            2. 自回归循环 n_steps 次:
               a. 取最近 HS 帧嵌入 + 对应动作 → 预测下一步嵌入
               b. 把预测嵌入拼回历史序列
               c. 把候选动作中的下一步动作拼入动作序列
            3. 返回所有候选序列的完整推演结果
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)  # 已有历史帧数
        B, S, T = action_sequence.shape[:3]
        # 动作序列的前 H 步是已执行的历史动作，后面才是待推演的未来动作
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H  # 需要推演的步数

        # 编码初始帧（只取第一帧编码，因为所有候选序列共享同一个起点）
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        # 把初始嵌入复制 S 份，对应 S 条候选动作序列
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # 把 (B, S) 拍平成 (B*S)，每条候选序列独立推演
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # ── 自回归推演循环 ──
        HS = history_size
        for t in range(n_steps):
            # 编码当前动作
            act_emb = self.action_encoder(act)
            # 只取最近 HS 帧作为预测器的输入（滑动窗口）
            emb_trunc = emb[:, -HS:]          # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]      # (BS, HS, A_emb)
            # 预测下一步嵌入，只取最后一个位置的输出
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            # 把预测结果拼回历史
            emb = torch.cat([emb, pred_emb], dim=1)

            # 取出候选动作序列中下一步的动作，拼入动作历史
            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)

        # 最后再预测一步（推演完所有未来步之后的最终状态）
        act_emb = self.action_encoder(act)
        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        # 恢复 (B*S) → (B, S)，每条候选序列对应一条推演轨迹
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """
        计算每条候选动作序列的代价：预测最终嵌入 vs 目标嵌入的 MSE。

        输入:
            info_dict["predicted_emb"]: (B, S, T, D) — 推演出的嵌入序列
            info_dict["goal_emb"]:      (B, S, T, D) — 目标状态的编码嵌入

        输出: cost (B, S) — 每条候选序列的代价，越小越好
        """
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # 只比较最后一步的嵌入距离（MPC 只看终点是否接近目标）
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),  # detach: 目标嵌入不参与梯度
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """
        MPC 规划入口：给定当前观测 + 目标 + 候选动作，返回每条候选的代价。

        调用链: encode(goal) → rollout(action_candidates) → criterion()
        """

        assert "goal" in info_dict, "goal not in info_dict"

        # 数据迁移到模型所在设备
        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        # 提取目标帧并编码
        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)  # 目标图像 → 目标嵌入

        info_dict["goal_emb"] = goal["emb"]
        # 自回归推演所有候选动作序列的未来轨迹
        info_dict = self.rollout(info_dict, action_candidates)
        # 计算每条候选序列的代价
        cost = self.criterion(info_dict)

        return cost
