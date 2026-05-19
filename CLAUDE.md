# CLAUDE.md

LeWorldModel (LeWM) — a minimal, two-loss JEPA that trains stably end-to-end from pixels.

## Commands

```bash
# Training (PushT is the default)
python train.py data=pusht                          # push T-block to target
python train.py data=pusht model.predictor.depth=4  # lighter predictor

# Evaluation (after training)
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# Download data (HuggingFace)
tar --zstd -xvf pusht_expert_train.tar.zst -C $STABLEWM_HOME/

# Environment setup
uv venv --python=3.10 && source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Data goes to `$STABLEWM_HOME` (default `~/.stable-wm/`). Dataset names in configs omit `.h5` extension.

## Architecture (4 files, ~400 lines of core logic)

```
train.py          Entry point. Hydra configs → dataset → model → Lightning training loop
jepa.py           JEPA class: encode(), predict(), rollout(), criterion(), get_cost()
module.py         Custom modules: ARPredictor, SIGReg, MLP, Embedder, Transformer variants
utils.py          Image preprocessing, Z-score normalization, checkpoint saving
config/train/     Hydra config hierarchy: lewm.yaml → model/lewm.yaml + data/{env}.yaml
config/eval/      Evaluation configs: environment + solver settings per task
```

### Data Flow (Training)

```
pixels (B, T, C, H, W) + action (B, T-1, act_dim)
  → encode(): flatten time → ViT per frame → CLS token → projector MLP → emb (B, T, 192)
  → split: ctx_emb[:,:3], tgt_emb[:,1:], ctx_act[:,:3]
  → predict(ctx_emb, ctx_act): ARPredictor (causal, AdaLN-zero conditioned on action)
  → loss = MSE(pred, tgt) + 0.09 * SIGReg(all_emb)
```

### Data Flow (Inference / MPC)

```
current obs + goal image
  → encode both → emb_0, goal_emb
  → sample 64 action sequences × T steps
  → rollout each: autoregressive predict in embedding space (no new images)
  → pick action sequence whose final predicted embedding is closest to goal_emb
  → execute first action, replan
```

## Key Design Decisions

**No EMA, no stop-gradient, no auxiliary losses.** The only loss terms are:
1. `pred_loss`: MSE between predicted embedding and encoder output for the same frame
2. `sigreg_loss`: Epps-Pulley statistic forcing embedding distribution → N(0,I), weight 0.09

**SIGReg is load-bearing.** Without it, the encoder collapses to a constant vector. The weight 0.09 is the only tuned hyperparameter — it balances "prevent collapse" vs "preserve semantic structure."

**AdaLN-zero conditioning in ARPredictor.** Action vectors modulate each Transformer layer via learned scale/shift parameters. Gates initialized to 0 → model learns to use actions gradually. This is more stable than concatenating actions to the input.

**Separate projector and pred_proj MLPs.** Even though input_dim == output_dim (both 192), these MLPs (192→2048→192 with BatchNorm) serve as adapter layers between the encoder's representation space and the predictor's. Removing them hurts convergence.

**History size of 3 is sufficient for these tasks.** The `ARPredictor` uses causal attention over the 3-frame context window. For tasks requiring velocity estimation (driving), consider 5-8.

## Hyperparameters That Matter

| Parameter | Default | When to Tune |
|-----------|---------|--------------|
| `sigreg.weight` | 0.09 | Only mandatory tune per new environment |
| `embed_dim` | 192 | Increase for visually complex scenes (256-384) |
| `history_size` | 3 | Increase for tasks requiring velocity estimation |
| `predictor depth` | 6 | Deeper for longer-horizon dynamics |
| `lr` | 5e-5 | Sensitive; 1e-4 often diverges |
| `batch_size` | 128 | Reduce if OOM (min 64 for stable SIGReg stats) |

## Training Health Checks

- **SIGReg loss dropping to 0** → collapse. Increase `sigreg.weight`.
- **SIGReg loss spiking and staying high** → over-regularized. Decrease `sigreg.weight`.
- **pred_loss plateaus high** → predictor underpowered. Increase depth or embed_dim.
- **val/pred_loss diverges from train** → overfitting. Increase dropout or reduce model size.
- **NaN in batch["action"]** → sequence boundaries in dataset. Code handles with `nan_to_num`.

## Files You Should NOT Modify

- `jepa.py` — the core JEPA logic (encode/predict/rollout/criterion). This is the algorithm.
- `module.py:10-36` — SIGReg implementation. The math is fragile; only tune `knots`/`num_proj` via config.

## Files You SHOULD Modify When Adapting to New Tasks

- `config/train/data/{new_task}.yaml` — dataset config (name, frameskip, keys_to_load)
- `config/train/lewm.yaml` — training hyperparams (lr, batch_size, embed_dim, history_size)
- `config/eval/{new_task}.yaml` — evaluation config (env, solver, goal_offset, eval_budget)
- `train.py` — add custom transforms/normalizers for new data columns
- `utils.py` — add custom preprocessing if needed
