
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

---

## How LeWM Works

### Core Principle: Predict Embeddings, Not Pixels

Traditional world models (e.g., Dreamer) reconstruct raw pixels frame-by-frame. This forces the model to waste capacity on irrelevant details — wall textures, lighting noise, static backgrounds. JEPA flips this: the **encoder** compresses each image into a compact embedding vector (192-dim) that only retains what matters for the future, and the **predictor** forecasts the next embedding directly. At no point does the model generate pixels.



### Architecture

```
                    Training                           │              Inference (MPC)
                                                       │
  pixels (B,T,C,H,W)                                   │   current obs + goal image
       │                                               │         │
       ▼                                               │         ▼
  ┌──────────┐    ┌──────────┐                         │   encode(obs)  encode(goal)
  │  ViT     │    │  ViT     │  ← same encoder          │     │              │
  │ (shared) │    │ (shared) │    no EMA, no stop-grad  │     ▼              ▼
  └────┬─────┘    └────┬─────┘                         │   emb_0         goal_emb
       │               │                               │     │
       ▼               ▼                               │     ▼
  ┌──────────┐    ┌──────────┐                         │   ┌──────────────────────┐
  │projector │    │projector │  ← MLP(192→2048→192)    │   │ Sample 64 action     │
  └────┬─────┘    └────┬─────┘                         │   │ sequences (T steps)  │
       │               │                               │   └─────────┬────────────┘
       ▼               ▼                               │             │
  emb (B,T,192)   tgt_emb  ──→ MSE(pred, tgt)          │   ┌─────────▼────────────┐
       │                                              │   │ ARPredictor rollout │
       ▼                                              │   │ (autoregressive)    │
  ┌──────────────────────────┐                        │   │ emb_0 → emb_1 → ... │
  │     ARPredictor          │                        │   └─────────┬────────────┘
  │  causal transformer ×6   │                        │             │
  │  AdaLN-zero condition    │←── action_emb          │             ▼
  └────────────┬─────────────┘                        │   ┌──────────────────────┐
               │                                      │   │ criterion:           │
               ▼                                      │   │ min MSE(pred[-1],    │
         pred_emb (B,T,192)                           │   │         goal_emb)    │
               │                                      │   └─────────┬────────────┘
               ▼                                      │             │
   ┌──────────────────────┐                           │             ▼
   │ pred_proj (MLP)      │                           │   Execute best action
   └──────────────────────┘                           │   step 0, replan
                                                      │
   Loss = MSE(pred, tgt) + 0.09 × SIGReg(emb)
```

Key components (all paths relative to repo root):

| Component | File | Role |
|-----------|------|------|
| `JEPA` | `jepa.py:11` | Top-level class: encode → predict → rollout → criterion |
| `ARPredictor` | `module.py:244` | Causal Transformer with AdaLN-zero conditioning |
| `SIGReg` | `module.py:10` | Epps-Pulley Gaussian regularizer (prevents collapse) |
| `MLP` | `module.py:217` | Shared projection head (projector / pred_proj) |
| `Embedder` | `module.py:189` | Action encoder (Conv1d + MLP) |
| ViT encoder | `stable_pretraining` | HuggingFace-style ViT (tiny, patch=14, img=224) |
| `lejepa_forward` | `train.py:17` | Per-batch forward + loss computation |

### The Collapse Problem and SIGReg

**Problem:** If the model only had the MSE prediction loss, the encoder would learn to output a constant vector (e.g., all zeros) — making the prediction loss trivially zero. This is **representation collapse**.

**Solution — SIGReg (Sketch Isotropic Gaussian Regularizer):** Forces the embedding distribution toward a standard Gaussian N(0, I). If embeddings must be Gaussian-distributed, they cannot all collapse to the same point.

Mechanism: project embeddings onto 1024 random directions, then check whether the projected values match the characteristic function of N(0, I): φ(t) = exp(-t²/2). This uses the Epps-Pulley test statistic over 17 evaluation points.

**Why 0.09?** This is the only hyperparameter you tune:
- Too large → embeddings are forced into a rigid Gaussian, losing semantic structure
- Too small → collapse is not prevented
- 0.09 is the equilibrium point validated across PushT, Cube, TwoRoom, and Reacher tasks

---

## Training Process — Step by Step

### Data Format

Each batch contains a contiguous sequence of frames + actions from pre-collected expert demonstrations (HDF5 format):

```
pixels:  (B, T_total, C, H, W)    e.g., (128, 4, 3, 224, 224) — 4 consecutive frames
action:  (B, T_total-1, act_dim)  e.g., (128, 3, 2)          — 3 actions between frames
```

Where `T_total = history_size + num_preds = 3 + 1 = 4`.

### One Forward Pass (`lejepa_forward` in `train.py:17`)

**Step 1 — Encode all frames independently** (`jepa.py:29`)

```python
pixels (B, 4, C, H, W)
  → rearrange to (B*4, C, H, W)   # flatten time into batch
  → ViT per frame                  # each frame gets its own CLS token
  → projector MLP                  # 192 → 2048 → 192
  → rearrange to (B, 4, 192)      # restore time dimension
```

Actions go through `action_encoder` (Conv1d kernel=1 + MLP) → `act_emb: (B, 3, 192)`.

**Step 2 — Split context and target**

```
ctx_emb = emb[:, :3]     # frames 0, 1, 2 — what the predictor sees
tgt_emb = emb[:, 1:]     # frames 1, 2, 3 — what the predictor should predict
ctx_act = act_emb[:, :3] # actions 0, 1, 2
```

The alignment: predictor sees frame `t` + action `t`, predicts frame `t+1`'s embedding.

**Step 3 — Predict** (`jepa.py:47` → `module.py:276`)

The `ARPredictor` is a 6-layer causal Transformer with AdaLN-zero conditioning:
- Action embeddings modulate each layer via learned scale/shift parameters
- Causal masking ensures position `t` only attends to `≤t`
- Gate parameters initialized to 0 → model first learns to ignore actions, then gradually incorporates them

```
ARPredictor.forward(x=ctx_emb, c=ctx_act):
  x += pos_embedding        # inject learned positional encoding
  for each ConditionalBlock:
    shift, scale, gate = adaLN_modulation(c)  # 6 params from action
    x += gate * attn(modulate(LN(x), shift, scale))  # causal attention
    x += gate * mlp(modulate(LN(x), shift, scale))   # FFN
  return x                  # (B, 3, 192) — predicted next embeddings
```

**Step 4 — Compute loss** (`train.py:38-41`)

```python
pred_loss   = (pred_emb - tgt_emb).pow(2).mean()   # prediction error
sigreg_loss = self.sigreg(emb.transpose(0, 1))      # Gaussian regularizer
loss        = pred_loss + 0.09 * sigreg_loss         # final loss
```

Note: `tgt_emb` comes from the **same encoder**, with **no stop-gradient**, **no EMA**, and **no auxiliary loss terms**. The SIGReg alone prevents collapse.

### Training Hyperparameters

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| `embed_dim` | 192 | Compact enough to prevent trivial solutions, large enough for rich semantics |
| `history_size` | 3 | 3 frames suffice for short-horizon dynamics; longer = more memory |
| `num_preds` | 1 | Single-step prediction; multi-step tested but 1-step works best |
| `predictor depth` | 6 | Shallow enough for fast training, deep enough for causal reasoning |
| `lr` | 5e-5 | Low lr prevents embedding space drift |
| `weight_decay` | 1e-3 | Light regularization on ViT backbone |
| `batch_size` | 128 | Balances GPU memory and gradient stability |
| `gradient_clip` | 1.0 | Prevents SIGReg gradient spikes during early training |
| `precision` | bf16 | 2× speedup over fp32, no stability issues |
| `epochs` | 100 | ~3-6 hours on a single GPU |

### Training Curves to Watch

- **pred_loss**: Should decrease steadily. If it plateaus high, the predictor capacity may be insufficient.
- **sigreg_loss**: Should stabilize around a small positive value. If it drops to zero, collapse is happening (increase λ). If it spikes and stays high, embeddings are being forced too hard (decrease λ).
- **val/pred_loss**: Should track training loss without a significant gap.

---

## Inference & Planning — Step by Step

LeWM is used for **Model Predictive Control (MPC)** at inference time. The trained model serves as a "world simulator" that imagines future states in embedding space.

### MPC Loop (`jepa.py:128` → `eval.py`)

For each environment step:

1. **Encode current observation**: `encode(pixels)` → `emb_0: (1, 192)`
2. **Encode goal image**: `encode(goal)` → `goal_emb: (1, 192)`
3. **Sample action candidates**: 64 random action sequences, each with `T` steps (e.g., T=16)
4. **Rollout** each candidate (see below) → `predicted_emb: (1, 64, T, 192)`
5. **Score** each candidate: `MSE(predicted_emb[...,-1:], goal_emb)` → cost `(1, 64)`
6. **Pick** the best candidate, execute its first action
7. **Replan** from the new observation (go to step 1)

### Rollout (`jepa.py:61`)

The rollout is purely in embedding space — no new images are encoded:

```python
emb = encode(current_pixels)                    # (1, 192) — initial state
for t in range(n_steps):                         # e.g., 16 future steps
    # Use last history_size=3 embeddings as context window
    context = emb[:, -3:]                        # sliding window
    # Encode the action at this step
    act_emb = action_encoder(action[:, t])
    # Predict next embedding
    next_emb = predictor(context, act_emb)[:, -1:]  # take last position
    # Append to history
    emb = cat([emb, next_emb])
# Final comparison: is the last predicted embedding close to goal_emb?
```

### CEM Solver (Optional but Recommended)

The default evaluation uses **Cross-Entropy Method (CEM)** instead of pure random sampling:
1. Sample 64 random action sequences
2. Evaluate all, keep top-k (e.g., 16)
3. Fit a Gaussian to the top-k sequences
4. Resample 64 new sequences from this Gaussian
5. Repeat 3-5 iterations → progressively refines toward the optimal plan

This typically yields 2-3× better planning performance than single-pass random sampling.

---

## Learning Path: PushT → CarRacing → CARLA → Real Robot

A progressive curriculum for mastering JEPA-based world models, ordered by visual complexity and data requirements.

### Stage 1: PushT (Push T-block to Target)

**Why first:** Fully configured in this repo. Single GPU, 3-6 hours. Produces visible results.

**What you learn:**
- End-to-end training pipeline (data → model → checkpoint → eval)
- How SIGReg prevents collapse (visualize embedding distributions)
- MPC planning with rollout and CEM

**Metrics to track:** pred_loss curve, sigreg_loss curve, planning success rate

**Data:** Pre-collected expert demonstrations (HDF5). Download from HuggingFace.

**Estimated effort:** 1-2 days (mostly waiting for training)

### Stage 2: CarRacing-v3 (Gymnasium / Box2D)

**Why second:** Slightly more complex visuals (road, car shape, track markings), continuous control (steering + throttle + brake), longer horizons. Same offline-training paradigm, but you need to collect your own data.

**What you learn:**
- Building a custom data collection pipeline (run a policy, record frames+actions to HDF5)
- Adapting LeWM to a new environment (write a data config YAML, tune embed_dim / history_size)
- Dealing with partially observable dynamics (car velocity matters, but isn't in the image directly)

**Key differences from PushT:**
- `action_dim`: 3 (steer, gas, brake) vs 2 in PushT
- `history_size`: consider 5-8 (velocity estimation needs more frames)
- `embed_dim`: consider 256-384 (more visual diversity)
- Data: need 500-2000 expert trajectories (use a scripted policy or RL-pretrained agent)

**Estimated effort:** 3-5 days (build data pipeline + train + tune)

### Stage 3: CARLA (Photorealistic Driving Simulator)

**Why third:** Photorealistic rendering, complex traffic scenarios, multi-agent dynamics. This is where JEPA's "predict in latent space" advantage becomes critical — pixel-level prediction would be infeasible.

**What you learn:**
- Handling high-resolution, photorealistic input (ViT-base or larger, 384×384 images)
- Multi-modal prediction (other vehicles behave unpredictably — the embedding must represent uncertainty)
- Long-horizon planning (driving decisions unfold over seconds, not frames)
- Data augmentation and domain randomization

**Key differences from CarRacing-v3:**
- Encoder: upgrade from ViT-tiny → ViT-small or ViT-base
- Image size: 224 → 384 or 448
- `embed_dim`: 384-768
- `history_size`: 10-16 (driving dynamics are slower)
- Data: use CARLA's autopilot or collect human driving data
- Consider adding auxiliary tasks (lane-keeping, speed prediction via probing)

**Challenges to expect:**
- Training time: 12-48 hours on a single GPU
- SIGReg may need tuning for the larger embed_dim
- The embedding space may need multi-step prediction (num_preds > 1) for stable long-horizon planning

**Estimated effort:** 1-3 weeks

### Stage 4: Real Robot World Model

**Why last:** Real-world data has noise, distribution shift, and safety constraints. This is the ultimate test of whether your learned world model generalizes.

**What you learn:**
- Domain gap between simulation and reality
- Fine-tuning strategies (pretrain in sim, adapt on real data)
- Safety-critical planning (cost shaping, constraint handling)
- Real-time inference optimization (model quantization, TensorRT)

**Key differences from CARLA:**
- Data: real robot teleoperation data (expensive to collect, high-value)
- Pretraining: use CARLA-trained weights as initialization
- Inference speed: may need model distillation or pruning for real-time control loops (>10 Hz)
- Evaluation: success rate alone isn't enough — need surprise detection, uncertainty quantification

**Prerequisites before attempting:**
- Stages 1-3 completed
- Reliable data collection infrastructure (teleoperation rig, synchronized cameras)
- Safety fallback policies (rule-based or simple PID backup)

**Estimated effort:** 4-8 weeks (heavily dependent on robot platform maturity)

### Stage Progression Summary

| Stage | Visual Complexity | Training Time | Data Required | Key Skill Learned |
|-------|------------------|---------------|---------------|-------------------|
| PushT | Low (solid shapes) | 3-6 hrs | Pre-collected | Core pipeline mastery |
| CarRacing-v3 | Medium (textured) | 6-12 hrs | 500-2000 trajs | Custom data pipeline |
| CARLA | High (photorealistic) | 12-48 hrs | 1000-5000 trajs | Scale & long-horizon |
| Real Robot | High + noise + shift | Days | 100-1000 demos | Sim-to-real transfer |

---

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
