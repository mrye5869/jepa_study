"""
LeWM 训练入口。

训练流程概览:
  1. 加载离线数据集（HDF5 格式，预录的专家轨迹）
  2. 用 Hydra 配置实例化 JEPA 模型 + SIGReg 正则器
  3. 包装为 PyTorch Lightning Module
  4. 启动训练循环

唯一的训练 forward 函数是 lejepa_forward，它定义了两项 loss:
  - pred_loss:   预测嵌入 vs 真实编码嵌入 的 MSE
  - sigreg_loss: 嵌入分布 vs 标准高斯分布 的 Epps-Pulley 统计量
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """
    LeWM 的完整前向传播 + loss 计算。

    这是训练时每个 batch 调用的核心函数，被包装为 Lightning Module 的 forward。

    步骤:
      1. encode:  所有帧逐个通过 ViT → 嵌入向量 (B, T, 192)
      2. 切分:    前 ctx_len 帧作为上下文，后 n_preds 帧作为目标
      3. predict: 上下文嵌入 + 动作 → 预测下一帧嵌入
      4. loss:    pred_loss（MSE 预测误差）+ λ × sigreg_loss（高斯正则）

    超参（来自 cfg）:
      - ctx_len (history_size=3): 用多少帧历史来预测
      - n_preds (num_preds=1):     预测几步后的嵌入
      - lambd (sigreg.weight=0.09): SIGReg 的权重 —— 这是唯一需要调的超参
    """

    ctx_len = cfg.wm.history_size   # 3 — 上下文帧数
    n_preds = cfg.wm.num_preds      # 1 — 预测步数
    lambd = cfg.loss.sigreg.weight  # 0.09 — SIGReg 权重

    # 序列边界处可能有 NaN（数据切分导致），替换为 0
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    # ── 第 1 步：编码所有帧 (+ 动作) ──
    # 输入: pixels (B, T_total, C, H, W), action (B, T_total-1, act_dim)
    # 输出: emb (B, T_total, 192) — 每一帧的嵌入
    #       act_emb (B, T_total-1, 192) — 每个动作的嵌入
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    # ── 第 2 步：切分上下文和目标 ──
    # 前 ctx_len=3 帧作为历史上下文，对应 ctx_len 个动作
    ctx_emb = emb[:, :ctx_len]       # 帧 0,1,2
    ctx_act = act_emb[:, :ctx_len]   # 动作 0,1,2

    # 目标是从第 1 帧开始的后续帧（跳过 n_preds 帧）
    # 即用帧 0,1,2 和动作 0,1,2 预测帧 1,2,3（时序对齐）
    tgt_emb = emb[:, n_preds:]       # 目标：帧 1,2,3 的真实嵌入

    # ── 第 3 步：预测 ──
    # ARPredictor 拿到上下文嵌入 + 动作，自回归预测下一步嵌入
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    # ── 第 4 步：计算两项 loss ──
    # 4a. 预测 loss: 预测嵌入与真实编码嵌入的 MSE
    #     注意：tgt_emb 来自同一编码器，没有 stop-grad、没有 EMA、没有额外约束
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()

    # 4b. SIGReg loss: 强制所有嵌入的分布接近标准高斯 N(0,I)
    #     这是防止表示坍缩的唯一约束
    #     transpose: (B, T, D) → (T, B, D)，因为 SIGReg 期望 (时间, 批次, 维度)
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    # 最终 loss = 预测误差 + 0.09 × 高斯正则
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # 记录各项 loss 到日志（WandB / TensorBoard）
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    """
    训练主函数。Hydra 会自动加载 config/train/lewm.yaml 及其引用的子配置。
    配置文件链:
      lewm.yaml → model/lewm.yaml (模型结构)
               → data/pusht.yaml   (数据集)
               → launcher/local.yaml (分布式配置)
    """

    # ═══════════════════════════════════════════
    # 1. 加载数据集
    # ═══════════════════════════════════════════

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    # 从 HDF5 文件加载离线数据集（预录的专家轨迹）
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    # 图像预处理：转 0-1 + ImageNet 归一化 + Resize 到 224×224
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]

    with open_dict(cfg):
        # 动作等非像素列做 Z-score 归一化
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        # 动作维度的计算：frameskip × 原始动作维度
        # 例如 frameskip=5 表示数据集每 5 帧记录一次动作，训练时用 1 步代表 5 步
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # 切分训练集和验证集（9:1）
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # DataLoader: batch_size=128, prefetch_factor=3, persistent_workers
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    # ═══════════════════════════════════════════
    # 2. 实例化模型
    # ═══════════════════════════════════════════

    # Hydra 根据 config/train/model/lewm.yaml 自动构建 JEPA 对象
    # 组件: ViT(encoder) + ARPredictor(predictor) + Embedder(action_encoder) + MLP×2
    world_model = hydra.utils.instantiate(cfg.model)

    # 优化器配置
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),           # AdamW, lr=5e-5, wd=1e-3
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)

    # 包装为 Lightning Module:
    #   - model=JEPA 对象
    #   - sigreg=SIGReg(knots=17, num_proj=1024)
    #   - forward=lejepa_forward（上面定义的函数）
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    # ═══════════════════════════════════════════
    # 3. 启动训练
    # ═══════════════════════════════════════════

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    # WandB 日志（可选）
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # 每个 epoch 保存一次 checkpoint
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    # PyTorch Lightning Trainer
    # 配置来自 lewm.yaml 的 trainer 字段: max_epochs=100, accelerator=gpu, precision=bf16
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    # Manager 负责断点续训：如果 checkpoint 存在则加载继续
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    # 开始训练
    manager()
    return


if __name__ == "__main__":
    run()
