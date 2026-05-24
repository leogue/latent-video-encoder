# Latent World Model

A **latent world model for robotics**, built in three stages around a V-JEPA-style
self-supervised video encoder — everything happens in latent space, with no pixel
reconstruction and no decoder.

| Stage | Package | What | Status |
|-------|---------|------|--------|
| 1 — **Encoder** | `lve` | Self-supervised video encoder (V-JEPA): predict masked spatio-temporal regions in latent space. Produces the frozen backbone. | ✅ working, validated on real robot video |
| 2 — **World predictor** | `lwp` | Action-conditioned dynamics (V-JEPA 2-AC): from frozen latents + state + action, predict the next-step latent. L1 regression, teacher-forcing + rollout. Block-causal transformer with separate visual/state/action input heads. | 🟡 core implemented & tested; real-data validation pending |
| 3 — **Planning** | `planning` | CEM / MPC over the predictor to reach a goal latent. | 🚧 planned |

The vision encoder and the dynamics are deliberately **modular**: a single frozen
encoder (Stage 1) is reused across many world-model / robot experiments. The clean
interface between stages is the exported encoder (`encoder.pt`).

The rest of this README documents **Stage 1 (the encoder)**, which is complete.
Stages 2–3 are under active development (`src/lwp`, `src/planning`).

---

## Idea in one picture

```
video clip  [B, C, T, H, W]
      │  tubelet patch embed (Conv3d)
      ▼
   tokens  [B, N, D]  ── multiblock mask ──►  context tokens   +   target tokens
                                                   │                    │
                                          context encoder θ      target encoder φ (EMA, no grad)
                                                   │                    │
                                              z_context              z_target  (stop-grad)
                                                   │                    │
                                                   ▼                    │
                                          predictor (z_context, target coords)
                                                   │                    │
                                                z_pred  ──── L1 loss ────┘   on LayerNorm(z_target)

θ (context encoder) + predictor  ← gradient
φ (target encoder)               ← EMA of θ:  φ ← m·φ + (1−m)·θ
```

The asymmetry (gradient student vs. stop-gradient EMA teacher) is what prevents
representation collapse — there is no contrastive term and no negative samples.

---

## Why these choices

Every default follows the V-JEPA / V-JEPA 2 recipe. Rationale and SOTA
references live in [`docs/DESIGN.md`](docs/DESIGN.md); the short version:

| Choice | What | Why |
|--------|------|-----|
| **Latent prediction, no decoder** | loss compares predicted vs. EMA-target *latents* | learns semantics, not pixels; the world-model goal needs representations, not reconstructions |
| **3D RoPE** | rotary position applied to Q/K per (t, y, x) axis | lets the encoder **extrapolate to longer clips at inference** without retraining position embeddings |
| **Multiblock masking** | short-range (8×, scale 0.15) + long-range (2×, scale 0.7) blocks, full temporal extent, ~90% masked | forces motion/semantic understanding; matches V-JEPA 2 |
| **Batch-shared mask** | one mask layout reused across the batch | keeps `N_context` / `N_target` fixed so they stack into `[B, N]` |
| **L1 on LayerNorm targets** | parameter-free LayerNorm over feature dim, then L1 | discourages dimensional collapse (not L2/`F.normalize`) |
| **EMA teacher** | momentum 0.998 → 1.0 (linear ramp) | the anti-collapse mechanism; V-JEPA 2 alternative = fixed momentum |
| **Dense tokens, no CLS** | full token grid returned | the downstream world model consumes per-token features |
| **Trapezoidal LR** | warmup → constant → cooldown | decouples LR from total length; easy to extend runs (V-JEPA 2) |

---

## Architecture

Three components (`src/lve/models/`):

- **`VideoTransformerEncoder`** — tubelet `VideoPatchEmbed` (Conv3d) → pre-norm
  Transformer blocks with 3D-RoPE attention. Returns dense tokens, grid shape and
  per-token `(t, y, x)` coordinates. Owns its patch embedding.
- **EMA target encoder** — a full copy of the context encoder (patch embed
  included), updated by EMA, never by gradient. *Not* a shared patch embed —
  that would break the EMA semantics.
- **`LatentPredictor`** — narrow Transformer: projects context latents, appends
  learned query tokens at the target positions, predicts target latents. Thrown
  away after pretraining.

Cross-cutting invariant: token flattening is **t-major, then y, then x**
(`index = t·Hg·Wg + y·Wg + x`); the Conv3d flatten, coordinate generation and
mask indices all agree on this order.

```
src/lve/
├── models/        patch_embed · pos_embed (RoPE/sincos) · transformer · encoder · predictor · jepa
├── masking.py     MultiBlockMaskCollator (batch-shared, context = complement of targets)
├── losses.py      L1 / MSE / SmoothL1 / Cosine on normalized targets
├── ema.py         init + update_ema + momentum schedule
├── optim.py       AdamW param groups (no-decay) · LR & weight-decay schedules
├── config.py      dataclasses + YAML loader + model factories
├── trainer.py     training loop, bf16/fp16 autocast, metrics, checkpointing
├── data/          synthetic · frame-folder (JPG) · video-folder datasets + clip-consistent transforms
└── probe.py       frozen-encoder attentive probe + exported-encoder loader
scripts/   train.py · export_encoder.py · eval_probe.py · preprocess_videos.py
configs/   tiny.yaml (dev/CPU) · real.yaml (frames) · vitl16.yaml (ViT-L reference)
tests/     patch_embed · rope · mask_generator · encoder_shapes · jepa_forward · ema
```

---

## Install

Uses [uv](https://docs.astral.sh/uv/) (Python ≥ 3.10):

```bash
uv sync          # installs torch, pyyaml, and the package (editable)
uv run pytest    # 25 tests
```

---

## Usage

### 1. Quick start (synthetic data, CPU)

```bash
uv run python scripts/train.py --config configs/tiny.yaml
```

Trains on procedurally generated clips — useful to check the pipeline runs. Note
that synthetic data is trivial and *will* collapse; it is a smoke test, not a
quality signal.

### 2. Train on real video

Video is decoded **offline** to JPG frames (torchvision 0.27 no longer decodes
video, and full-decoding long clips exhausts memory). Frames are then read with
torchvision's JPEG decoder — no extra video-decoder dependency.

```bash
# 1. Extract frames (native fps; control sampling rate via frame_stride)
uv run python scripts/preprocess_videos.py --src data/videos --dst data/frames --fps 0 --size 256

# 2. Train
uv run python scripts/train.py --config configs/real.yaml

# Resume
uv run python scripts/train.py --config configs/real.yaml --resume runs/real/checkpoints/latest.pt
```

The effective temporal rate is `native_fps / frame_stride` (e.g. 25/4 ≈ 6 fps,
2.56 s clips — the V-JEPA range). Prefer extracting at native fps and tuning
`frame_stride`, so you can re-tune the temporal span without re-extracting.

### 3. Export the encoder (the deliverable)

```bash
uv run python scripts/export_encoder.py --checkpoint runs/real/checkpoints/latest.pt --out encoder.pt
```

Exports the **EMA (target) encoder** as a self-contained artifact (weights +
model config + preprocessing). A consumer can rebuild it and call
`encoder.embed(video) → tokens [B, N, D] + grid_shape + coords` with no training
code.

### 4. Evaluate (downstream probe)

```bash
uv run python scripts/eval_probe.py --encoder encoder.pt --num-classes N
```

Trains an attentive probe on the **frozen** encoder. Plug a labeled clip dataset
into `build_labeled_dataset` (e.g. Something-Something v2, Kinetics-400,
Diving-48). This is the only reliable measure of representation quality.

---

## Data formats

| Backend | Input | Use |
|---------|-------|-----|
| `synthetic` | none (generated) | dev / CI smoke test |
| `frames` | `data/frames/<video>/frame_*.jpg` (via `preprocess_videos.py`) | **recommended** for loose video files |
| `lerobot` | a [LeRobotDataset](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) v2.1 / v3.0 | **robot data** (the field standard) |
| `torchvision` | folder of video files | legacy; broken on torchvision ≥ 0.27 (no video decoder) |

### LeRobot

The `lerobot` backend samples temporal clips from one camera stream
(`observation.images.<cam>`) of a real LeRobotDataset, using `delta_timestamps`.
Actions and low-dim state are ignored — the encoder trains on pixels only;
actions belong to the later action-conditioned stage.

```bash
uv add lerobot        # optional heavy dependency
uv run python scripts/train.py --config configs/lerobot.yaml
```

Set `data.repo_id` (Hub id or local dataset), optionally `data.root` (local path)
and `data.camera_key` (defaults to the first camera). This expects a real
LeRobotDataset (`meta/` + Parquet + MP4 shards), *not* loose video files — for
those use the `frames` backend.

---

## Monitoring & avoiding false signals

The training loss and `pred_target_cosine` are **not** quality signals — a
collapsed encoder drives both to look great. Watch instead:

- **`within-clip token std`** — near 0 ⇒ the encoder outputs a constant
  (collapsed); healthy values are orders of magnitude larger.
- **spread of pooled embeddings across clips** — should grow as the encoder
  learns to distinguish clips.

In a real run on robot video, the loss plateaued (~0.27) while these diagnostics
kept improving (cross-clip cosine 0.995 → 0.967, pooled spread 2×) — i.e. the
encoder kept learning while the loss said nothing.

---

## Model scales

| Config | encoder dim / depth | encoder params | Target hardware |
|--------|---------------------|----------------|-----------------|
| `tiny` / `real` | 384 / 6 | ~12M | CPU-trainable |
| ViT-S | 384 / 12 | ~22M | entry GPU |
| ViT-B | 768 / 12 | ~86M | RTX 3060-class (sweet spot) |
| `vitl16` | 1024 / 24 | ~304M | multi-GPU / tight on 12GB |

On GPU, set `device: cuda`, `precision: bf16`, `training.compile: true`, and
raise `batch_size`.

---

## Scope & limitations

- **The encoder (Stage 1) has no action conditioning, dynamics predictor, or
  decoder by design** — action-conditioned dynamics live in Stage 2 (`lwp`),
  which consumes the *frozen* encoder. Keeping vision and dynamics modular is
  intentional.
- **Causal/future masking is intentionally absent in the encoder**; full-temporal-extent
  masking is the correct encoder-stage recipe (V-JEPA 2). Block-causal,
  action-conditioned prediction belongs to Stage 2.
- Representation quality requires real, diverse data and a downstream probe;
  small same-scene datasets adapt to one environment but won't yield a generalist
  encoder.

---

## References

1. Bardes et al., **V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video** (Meta AI, 2024). [blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) · [code](https://github.com/facebookresearch/jepa)
2. Assran et al., **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** (Meta AI, 2025). [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) · [code](https://github.com/facebookresearch/vjepa2)
3. **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning** (2026). [arXiv:2603.14482](https://arxiv.org/abs/2603.14482)
4. Su et al., **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021). [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

Full design rationale and the annotated spec: [`docs/DESIGN.md`](docs/DESIGN.md).
