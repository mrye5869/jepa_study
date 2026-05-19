# LeWM — 从像素端到端训练的世界模型

基于 JEPA（联合嵌入预测架构）的极简世界模型。只用两项 loss（预测误差 + 高斯正则），无 EMA、无 stop-gradient、无辅助监督。15M 参数，单 GPU 数小时可训完，规划速度比基于 foundation model 的世界模型快 48 倍。

论文: [arxiv.org/pdf/2603.19312v1](https://arxiv.org/pdf/2603.19312v1) | 数据: [HuggingFace](https://huggingface.co/collections/quentinll/lewm) | [项目主页](https://le-wm.github.io/)

```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 目录

- [环境安装与数据准备](#环境安装与数据准备)
- [代码结构速览](#代码结构速览)
- [训练流程标准](#训练流程标准)
- [推理流程标准（MPC 规划）](#推理流程标准mpc-规划)
- [学习计划：PushT → CarRacing → CARLA → 真实机器人](#学习计划pusht--carracing--carla--真实机器人)
- [超参与训练诊断](#超参与训练诊断)
- [适配新任务](#适配新任务)

---

## 环境安装与数据准备

```bash
# 环境
uv venv --python=3.10 && source .venv/bin/activate
uv pip install stable-worldmodel[train,env]

# 数据（PushT 为例）
tar --zstd -xvf pusht_expert_train.tar.zst -C $STABLEWM_HOME/
```

数据文件（`.h5` 格式）放在 `$STABLEWM_HOME`（默认 `~/.stable-wm/`）。可覆盖：

```bash
export STABLEWM_HOME=/你要的路径
```

配置中引用数据集时 **不加 `.h5` 后缀**。例如 `config/train/data/pusht.yaml` 里写 `pusht_expert_train`，实际文件为 `$STABLEWM_HOME/pusht_expert_train.h5`。

---

## 代码结构速览

四个核心文件，核心逻辑约 400 行：

| 文件 | 职责 | 关键函数/类 |
|------|------|------------|
| `train.py` | 训练入口，Hydra 加载配置 → 数据 → 模型 → Lightning 训练 | `lejepa_forward()` |
| `jepa.py` | JEPA 主类：编码、预测、推演、代价计算 | `encode()`, `predict()`, `rollout()`, `criterion()`, `get_cost()` |
| `module.py` | 自定义底层模块 | `ARPredictor`, `SIGReg`, `MLP`, `Embedder`, `ConditionalBlock` |
| `utils.py` | 图像预处理、Z-score 归一化、checkpoint 回调 | `get_img_preprocessor()`, `get_column_normalizer()` |

辅助目录：

```
config/train/             Hydra 训练配置：lewm.yaml → model/lewm.yaml + data/{任务}.yaml
config/eval/              评估配置：环境 + CEM 求解器 + 规划预算
```

### 架构全景图

```
                    训练                                    │              推理 (MPC)
                                                           │
 pixels (B,T,C,H,W) + action (B,T-1,act_dim)               │   当前观测 + 目标图像
      │                                                    │        │
      ▼                                                    │        ▼
 ┌──────────┐     ┌──────────┐                             │  encode(obs)   encode(goal)
 │  ViT ×T  │     │  ViT ×T  │  ← 同一编码器                │      │              │
 │ 逐帧独立  │     │ 逐帧独立  │    无 EMA，无 stop-grad     │      ▼              ▼
 └────┬─────┘     └────┬─────┘                             │   emb_0         goal_emb
      │                │                                   │      │
      ▼                ▼                                   │      ▼
 ┌──────────┐     ┌──────────┐                             │  ┌─────────────────────────┐
 │projector │     │projector │  ← MLP(192→2048→192)       │  │ 采样 64 条候选动作序列    │
 └────┬─────┘     └────┬─────┘                             │  └───────────┬─────────────┘
      │                │                                   │              │
      ▼                ▼                                   │  ┌───────────▼─────────────┐
 emb (B,T,192)    tgt_emb ───→ MSE(pred, tgt)             │  │ ARPredictor rollout     │
      │                                                    │  │ 自回归推演，不碰新图像    │
      ▼                                                    │  │ emb₀ → emb₁ → ... → embₙ│
 ┌────────────────────────────┐                            │  └───────────┬─────────────┘
 │       ARPredictor          │                            │              │
 │  因果 Transformer ×6       │                            │              ▼
 │  AdaLN-zero 条件注入       │←── action_emb               │  ┌─────────────────────────┐
 └─────────────┬──────────────┘                            │  │ criterion:              │
               │                                           │  │ min MSE(pred[-1], goal) │
               ▼                                           │  └───────────┬─────────────┘
         pred_emb (B,T,192)                                │              │
               │                                           │              ▼
               ▼                                           │  执行最优动作第 0 步，重新规划
   ┌────────────────────┐                                  │
   │ pred_proj (MLP)    │                                  │
   └────────────────────┘                                  │

   Loss = MSE(pred, tgt) + 0.09 × SIGReg(all_emb)
```

### SIGReg：防止表示坍缩的唯一防线

只保留 MSE 预测 loss 会出问题：编码器学到把所有图像映射到同一个常数向量（如全零），预测 loss 为零，但学到了空气。这就是**表示坍缩**。

SIGReg（Sketch Isotropic Gaussian Regularizer）强制嵌入分布接近标准高斯 N(0, I)。原理：把嵌入投影到 1024 个随机方向上，用 Epps-Pulley 检验检查投影值是否匹配标准高斯的特征函数 φ(t) = exp(-t²/2)。不匹配就惩罚。既然所有嵌入必须像高斯一样分散，就不可能挤到同一个点上。

权重 0.09 是唯一的可调超参：太大 → 强制高斯过于刚性，丧失语义结构；太小 → 坍缩拦不住。0.09 是 PushT/Cube/TwoRoom/Reacher 四个任务上验证的均衡点。

---

## 训练流程标准

### 第一步 — 准备数据

离线数据集：预录的专家演示轨迹，HDF5 格式。每次取连续 `history_size + num_preds` 帧。以 PushT 为例：

```
每个 batch:
  pixels: (128, 4, 3, 224, 224)   ← 连续 4 帧 (history_size=3 + num_preds=1)
  action: (128, 3, 2)             ← 帧间的 3 个动作
```

图像预处理：转 0-1 → ImageNet 均值/标准差归一化 → Resize 到 224×224。动作做 Z-score 归一化。序列边界处的 NaN 用 `nan_to_num` 置零。

代码路径：`train.py:47-79`（数据集加载 + 预处理）。

### 第二步 — 编码

把所有帧拍平到 batch 维，逐帧独立通过 ViT，取 CLS token，过 projector MLP。

```
pixels (B, 4, C, H, W)
  → rearrange → (B*4, C, H, W)     # 时间维拍平，每帧当作独立图像
  → ViT → CLS token → (B*4, 192)
  → projector MLP(192→2048→192)    # 映射到共享嵌入空间
  → rearrange → emb (B, 4, 192)    # 恢复时间维

action (B, 3, act_dim)
  → Embedder (Conv1d + MLP) → act_emb (B, 3, 192)
```

代码路径：`jepa.py:29-45`。

### 第三步 — 切分上下文与目标

```
ctx_emb = emb[:, :3]     # 帧 0,1,2 — 预测器的输入
tgt_emb = emb[:, 1:]     # 帧 1,2,3 — 预测器应输出的真实嵌入（标签）
ctx_act = act_emb[:, :3] # 动作 0,1,2 — 条件信号
```

时序对齐：预测器拿到帧 t 的嵌入 + 动作 t，应预测帧 t+1 的嵌入。训练时用 MSE 逐位比较。

代码路径：`train.py:30-36`。

### 第四步 — 预测

ARPredictor：6 层因果 Transformer，AdaLN-zero 条件注入。

```
ARPredictor.forward(x=ctx_emb, c=ctx_act):
  x += pos_embedding           # 可学习位置编码
  for 每个 ConditionalBlock:
    shift, scale, gate = adaLN_modulation(c)  # 从动作嵌入生成 6 个调制参数
    x += gate * attention(modulate(LN(x), shift, scale))  # 因果注意力
    x += gate * mlp(modulate(LN(x), shift, scale))        # 前馈网络
  return x  # (B, 3, 192)
```

AdaLN-zero 的关键：gate 初始化为 0，训练初期条件分支被关闭，模型先学"单靠视觉能预测多少"，再逐渐学会利用动作信号。因果掩码确保位置 t 只能 attend ≤t 的位置（自回归约束）。

代码路径：`jepa.py:47-55` → `module.py:244-285`（ARPredictor）→ `module.py:88-111`（ConditionalBlock）。

### 第五步 — 计算 Loss

```python
# train.py:38-45
pred_loss   = (pred_emb - tgt_emb).pow(2).mean()   # 预测误差
sigreg_loss = sigreg(emb.transpose(0, 1))            # 高斯正则
loss        = pred_loss + 0.09 * sigreg_loss          # 最终 loss
```

两点注意：
- `tgt_emb` 来自**同一个编码器**，无 stop-grad、无 EMA、无额外 trick
- SIGReg 统计的是所有嵌入（上下文 + 目标），而非仅预测值。约束的是编码器产出的分布本身

### 训练命令

```bash
# WandB 配置（config/train/lewm.yaml）
# 设置 entity 和 project，或关掉 wandb.enabled

# 启动训练
python train.py data=pusht

# 用更浅的预测器（减少参数量）
python train.py data=pusht model.predictor.depth=4
```

### 训练中需要盯的曲线

| 曲线 | 正常表现 | 异常信号 |
|------|---------|---------|
| `train/pred_loss` | 稳定下降 | 高平台 → 预测器容量不足，加深度或嵌入维度 |
| `train/sigreg_loss` | 稳定在小的正值 | 跌到零 → 坍缩，增大 λ；飙升不降 → 过正则，减小 λ |
| `val/pred_loss` | 紧跟训练 loss | 与训练 loss 分叉 → 过拟合，加 dropout 或减小模型 |

Checkpoint 每个 epoch 保存一次到 `$STABLEWM_HOME/checkpoints/`。

---

## 推理流程标准（MPC 规划）

训练好的 LeWM 是一个"世界模拟器"——给定当前状态和动作，预测未来状态嵌入。推理时用它做模型预测控制（MPC）。

### MPC 主循环

每步环境交互执行以下流程：

```
1. encode(当前帧) → emb_0 (1, 192)
2. encode(目标帧) → goal_emb (1, 192)
3. 采样 64 条候选动作序列，每条 T 步（如 T=16）
4. rollout: 每条候选序列自回归推演 T 步，得到预测嵌入轨迹
5. criterion: MSE(每条轨迹的最终嵌入, goal_emb) → 64 个代价
6. 选代价最小的序列，执行其第 1 个动作
7. 环境给出新观测 → 回到步骤 1（重新规划）
```

代码路径：`jepa.py:128-153`（`get_cost`）→ `eval.py:49-173`。

### Rollout 推演细节

Rollout 完全在嵌入空间进行——**不需要任何新图像**。

```
输入: emb_0 (1, 192)  + 候选动作序列 (1, 64, T, act_dim)

for t in range(推演步数):
    context = emb[:, -3:]                    # 取最近 3 帧嵌入（滑动窗口）
    act_emb = action_encoder(action[:, t])   # 编码当前步动作
    next_emb = predictor(context, act_emb)[:, -1:]  # 预测下一步嵌入
    emb = cat([emb, next_emb])               # 拼回历史

输出: predicted_emb (1, 64, T, 192)  ← 每条候选序列的推演轨迹
```

代码路径：`jepa.py:61-110`。

### CEM 求解器

默认评估使用交叉熵方法（CEM）迭代精化，比单次随机采样效果好 2-3 倍：

```
第 1 轮：随机采样 64 条动作序列 → 评估 → 保留 top-16
第 2 轮：用 top-16 拟合高斯分布 → 重新采样 64 条 → 评估 → 保留 top-16
...重复 3-5 轮
```

最终选代价最小的序列执行。代码路径：`config/eval/solver/cem.yaml`。

### 评估命令

```bash
# 用训练好的模型做评估
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# policy 是 $STABLEWM_HOME 下的相对路径，不加 _object.ckpt 后缀
```

---

## 学习计划：PushT → CarRacing → CARLA → 真实机器人

从零基础到能用 JEPA 做真实机器人世界模型的渐进路线，按视觉复杂度和数据需求递增排列。

### 第一阶段：PushT（推动 T 形块到目标）

**为什么先做这个：** 本仓库已完整配置，单 GPU 3-6 小时出结果。

**学到的能力：**
- 端到端训练管线（数据 → 模型 → checkpoint → 评估）
- 通过可视化嵌入分布理解 SIGReg 如何防止坍缩
- MPC 规划（rollout + CEM 求解）

**需要跟踪的指标：** pred_loss 曲线、sigreg_loss 曲线、规划成功率

**数据来源：** HuggingFace 下载预录专家数据

**预估时间：** 1-2 天（大部分时间在等训练）

---

### 第二阶段：CarRacing-v3（Gymnasium / Box2D）

**为什么第二个：** 视觉略复杂（赛道纹理、车体形状），连续控制（转向 + 油门 + 刹车），更长的时序范围。同样离线训练，但需要自己采集数据。

**学到的能力：**
- 构建自定义数据采集管线（运行策略 → 录制帧和动作 → 存为 HDF5）
- 适配 LeWM 到新环境（写数据配置 YAML，调 embed_dim / history_size）
- 处理部分可观测动力学（车速不能直接从单帧图像读出）

**与 PushT 的关键差异：**

| 项目 | PushT | CarRacing-v3 |
|------|-------|-------------|
| `action_dim` | 2（x, y 速度） | 3（转向、油门、刹车） |
| `history_size` | 3 | 5-8（速度估计需要更多帧） |
| `embed_dim` | 192 | 256-384（视觉多样性更大） |
| 数据量 | 预录 | 500-2000 条轨迹 |

**预估时间：** 3-5 天（搭数据管线 + 训练 + 调参）

---

### 第三阶段：CARLA（照片级驾驶模拟器）

**为什么第三个：** 照片级渲染、复杂交通场景、多智能体交互。JEPA"在嵌入空间预测"的优势在这里真正体现——逐像素预测完全不现实。

**学到的能力：**
- 处理高分辨率照片级输入（ViT-base 或更大，384×384 图像）
- 多模态预测（其他车辆行为不可预测 → 嵌入需要编码不确定性）
- 长时域规划（驾驶决策跨秒而非跨帧）
- 数据增强和域随机化

**与 CarRacing-v3 的关键差异：**

| 项目 | CarRacing-v3 | CARLA |
|------|-------------|-------|
| 编码器 | ViT-tiny | ViT-small 或 ViT-base |
| 图像尺寸 | 224 | 384 或 448 |
| `embed_dim` | 256-384 | 384-768 |
| `history_size` | 5-8 | 10-16（驾驶动力学更慢） |
| 数据来源 | 脚本策略 | CARLA 自动驾驶或人工驾驶数据 |
| 训练时间 | 6-12 小时 | 12-48 小时 |

**可能的坑：**
- SIGReg 在大嵌入维度下可能需要重新调权重
- 嵌入空间可能需要多步预测（`num_preds > 1`）来稳定长时域规划
- 考虑加辅助探测任务（车道保持、速度预测）

**预估时间：** 1-3 周

---

### 第四阶段：真实机器人世界模型

**为什么最后：** 真实数据有噪声、分布漂移、安全约束。这是学到的世界模型能否泛化的终极检验。

**学到的能力：**
- 仿真到现实的域差距处理
- 微调策略（仿真预训练 → 真实数据适配）
- 安全关键规划（代价塑形、约束处理）
- 实时推理优化（量化、TensorRT）

**与 CARLA 的关键差异：**

| 项目 | CARLA | 真实机器人 |
|------|-------|----------|
| 数据 | CARLA 自动驾驶 | 真实遥操作数据（昂贵、高价值） |
| 预训练 | 从零训练 | 用 CARLA 权重做初始化 |
| 推理速度 | 不限 | 可能需要蒸馏或剪枝（>10 Hz） |
| 评估 | 成功率 | 成功率 + 意外检测 + 不确定性量化 |

**前置条件：** 前三阶段全部完成 + 可靠的数据采集基础设施（遥操作装置 + 同步相机）+ 安全回退策略（规则或 PID 备份）

**预估时间：** 4-8 周（高度依赖机器人平台成熟度）

---

### 四阶段总览

| 阶段 | 视觉复杂度 | 训练时间 | 数据需求 | 核心能力 |
|------|-----------|---------|---------|---------|
| PushT | 低（纯色几何体） | 3-6 小时 | 预录数据 | 掌握核心管线 |
| CarRacing-v3 | 中（有纹理） | 6-12 小时 | 500-2000 条轨迹 | 自定义数据管线 |
| CARLA | 高（照片级） | 12-48 小时 | 1000-5000 条轨迹 | 规模化 + 长时域 |
| 真实机器人 | 高 + 噪声 + 漂移 | 数天 | 100-1000 次演示 | 仿真到现实迁移 |

---

## 超参与训练诊断

### 核心超参

| 参数 | 默认值 | 何时调 |
|------|-------|-------|
| `sigreg.weight` | 0.09 | 每个新环境必调 |
| `embed_dim` | 192 | 视觉复杂场景增大（256-384） |
| `history_size` | 3 | 需要速度估计的任务增大（5-8） |
| `predictor depth` | 6 | 更长时域动力学加深 |
| `lr` | 5e-5 | 敏感；1e-4 常导致发散 |
| `batch_size` | 128 | OOM 时减小（SIGReg 稳定至少需 64） |
| `gradient_clip` | 1.0 | 防止 SIGReg 梯度在训练早期爆炸 |

### 故障排查

| 症状 | 诊断 | 处理 |
|------|------|------|
| SIGReg loss 跌到 0 | 表示坍缩 | 增大 `sigreg.weight` |
| SIGReg loss 飙升不降 | 过正则 | 减小 `sigreg.weight` |
| pred_loss 高平台 | 预测器容量不足 | 加深或加宽 |
| val/pred_loss 与训练分叉 | 过拟合 | 加 dropout、减模型尺寸 |
| batch 中 action 出现 NaN | 数据集序列边界 | 代码已用 `nan_to_num` 处理 |

---

## 适配新任务

### 可以改的文件

- `config/train/data/{新任务}.yaml` — 数据集名称、`frameskip`、`keys_to_load`
- `config/train/lewm.yaml` — `lr`、`batch_size`、`embed_dim`、`history_size`
- `config/eval/{新任务}.yaml` — 环境、CEM 求解器参数、`goal_offset`、`eval_budget`
- `train.py` — 为新数据列添加自定义 transform / normalizer
- `utils.py` — 添加自定义预处理逻辑

### 不要改的文件

- `jepa.py` — JEPA 核心逻辑（encode/predict/rollout/criterion）。这是算法本身。
- `module.py:10-36` — SIGReg 实现。数学脆弱，只通过 config 调 `knots`/`num_proj`。
