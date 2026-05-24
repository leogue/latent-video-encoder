## Main architecture overview

The model has three neural components:

```text
1. context_encoder: VideoTransformerEncoder
2. target_encoder: VideoTransformerEncoder
3. predictor: LatentPredictorTransformer
```

The context encoder and target encoder have the same architecture.

The target encoder is initialized from the context encoder.

During training:

```text
context_encoder receives visible tokens
target_encoder receives full or target tokens
predictor receives context latents + target positions
predictor outputs predicted target latents
loss compares predicted target latents to target encoder latents
target encoder is updated with EMA, not gradient
```

---

## Scope and downstream contract

**This repo trains the video encoder only.** The eventual goal is a world model
for robotics (an action-conditioned predictor, à la V-JEPA 2-AC), but that is a
**separate, later stage** built on top of the *frozen* encoder produced here. Do
not add action conditioning, a dynamics/rollout predictor, or a decoder to this
repo. The JEPA predictor here exists solely as a pretraining crutch to train the
encoder — it is discarded after pretraining.

Because the downstream consumer is a world model, the encoder must expose **dense
spatio-temporal features**, not a pooled vector. The frozen-encoder contract the
AC stage will depend on:

```python
class VideoTransformerEncoder:
    def embed(self, video: torch.Tensor) -> EncoderOutput:
        """video [B, C, T, H, W] -> tokens [B, N, D], grid_shape (Tg,Hg,Wg),
        coords [N, 3]. Deterministic; no masking; full token grid."""
```

Requirements that protect the downstream use:
- `use_cls_token = False` — keep the full token grid.
- `coords` and the RoPE convention must be **reproducible at inference** for
  arbitrary clip length/resolution (the AC predictor re-embeds new observations).
- The shipped export must be self-contained (patch-embed + position config +
  tubelet/grid params), see the export section.

### Target scale

`configs/tiny.yaml` is for development/CI only. The encoder you actually ship for
robotics should be **ViT-L/16** (the V-JEPA 2 reference): `embed_dim 1024,
depth 24, num_heads 16`, tubelet `2×16×16`, 16 frames @ 224, `sampling_rate 4`,
predictor `depth 12, embed_dim 384`. Provide a `configs/vitl16.yaml` at that
scale. Bigger frozen encoder ⇒ better world model; favor encoder capacity over
predictor capacity.

### Temporal context: prefer span over density

A world model wants temporal context, but **do not get it by adding frames**.
Attention is O(N²) in tokens, `N = Tg·Hg·Wg`, so 16→32→64 frames costs ~1×→4×→16×.
Distinguish:

- **temporal span** (seconds covered) — controlled by `sampling_rate`, *free* in
  compute. 16 frames @ `sampling_rate 4` already span 64 raw frames (~2.5s @25fps).
  Want more context now? Raise `sampling_rate` (6–8), not `num_frames`.
- **temporal density** (frames/sec sampled) — controlled by `num_frames`,
  expensive; only worth it for fast fine-grained motion.

Default and ship at **16 frames** with `sampling_rate` as the tuning knob. 3D-RoPE
already lets the encoder **extrapolate to longer sequences at inference** than it
saw in training, so the world-model stage can feed longer clips without
retraining. Keep `num_frames`, `sampling_rate`, and tubelet temporal size as
first-class config knobs.

**Switching frame count (e.g. 8 ↔ 16) must be a config-only change**, no code
edits. This is free as long as:
- `num_frames` is divisible by the tubelet temporal size (`pt`): 8→Tg=4, 16→Tg=8.
- `data.clip_frames == model.num_frames` (assert at startup — the dataloader must
  emit exactly what the encoder expects).
- `pos_embed_type` is `rope_3d` (or `sinusoidal_3d`): coords/masks are derived
  from `grid_shape` at runtime, so nothing is hardcoded and **a checkpoint trained
  at 8 frames can even be reused at 16**. `learned_abs` breaks this (fixed-size
  `pos_embed` parameter) — another reason RoPE is the default.

> Deferred (do NOT build in v1): V-JEPA 2-style progressive-resolution / longer-clip
> cooldown phase. It needs a multi-phase data pipeline and position handling for
> changing token counts. Document it as a future phase; ship the fixed-resolution
> recipe first.

---

## Input video format

The dataloader should output video clips as:

```python
video: torch.Tensor
shape: [B, C, T, H, W]
dtype: float32
range: normalized
```

Example:

```python
B = 4
C = 3
T = 16
H = 224
W = 224
```

Shape:

```text
[4, 3, 16, 224, 224]
```

Use `[B, C, T, H, W]` internally because this is conventional in PyTorch video models.

---

## Tubelet patch embedding

The video is converted into spatio-temporal tubelet tokens.

Given:

```python
video.shape = [B, C, T, H, W]
tubelet_size = (pt, ph, pw)
```

Example:

```python
pt = 2
ph = 16
pw = 16
```

Then:

```text
T_tokens = T / pt
H_tokens = H / ph
W_tokens = W / pw
N = T_tokens * H_tokens * W_tokens
```

For:

```text
T = 16
H = 224
W = 224
pt = 2
ph = 16
pw = 16
```

we get:

```text
T_tokens = 8
H_tokens = 14
W_tokens = 14
N = 8 * 14 * 14 = 1568
```

Each tubelet contains:

```text
C * pt * ph * pw
```

values.

With RGB:

```text
3 * 2 * 16 * 16 = 1536
```

The tubelet is projected to model dimension `D`:

```python
raw_tubelet: [B, N, C * pt * ph * pw]
embedding:   [B, N, D]
```

Implement `VideoPatchEmbed`.

Use a Conv3d implementation for efficiency:

```python
nn.Conv3d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=(pt, ph, pw),
    stride=(pt, ph, pw),
)
```

The Conv3d output is:

```python
[B, D, T_tokens, H_tokens, W_tokens]
```

Then flatten:

```python
[B, N, D]
```

Also return the grid shape:

```python
grid_shape = (T_tokens, H_tokens, W_tokens)
```

---

## Position information

The model must support multiple position encoding modes.

Implement:

```python
pos_embed_type: Literal[
    "learned_abs",
    "sinusoidal_3d",
    "rope_3d",
    "none"
]
```

### 1. learned_abs

Learn a parameter:

```python
pos_embed: [1, N, D]
```

Then:

```python
tokens = tokens + pos_embed
```

This is simple but less flexible for resolution changes.

### 2. sinusoidal_3d

Generate deterministic position vectors from:

```text
time index
height index
width index
```

Use factorized allocation of dimensions:

```text
D_t + D_h + D_w = D
```

Then concatenate or sum projected embeddings.

### 3. rope_3d

Implement 3D Rotary Position Embedding.

RoPE should be applied to Q and K inside attention, not added to token embeddings.

Each token has integer coordinates:

```python
coords: [N, 3]
coords[:, 0] = t
coords[:, 1] = y
coords[:, 2] = x
```

Inside attention:

```python
q = apply_3d_rope(q, coords)
k = apply_3d_rope(k, coords)
```

Expected attention shapes:

```python
q: [B, num_heads, N, head_dim]
k: [B, num_heads, N, head_dim]
v: [B, num_heads, N, head_dim]
```

For 3D RoPE, split `head_dim` across axes:

```text
head_dim_t + head_dim_y + head_dim_x <= head_dim
```

A practical default:

```text
use 1/3 for time
use 1/3 for height
use 1/3 for width
leave remainder unrotated if needed
```

Implement cleanly.

Do not over-engineer.

Add tests verifying:

* output shape unchanged,
* RoPE does not modify V,
* RoPE works for variable sequence lengths,
* no NaNs.

---

## Transformer encoder block

Implement a standard pre-norm Transformer block:

```text
x = x + DropPath(SelfAttention(LayerNorm(x)))
x = x + DropPath(MLP(LayerNorm(x)))
```

Where:

```python
x: [B, N, D]
```

### Multi-head self-attention

Given:

```python
D = embed_dim
H = num_heads
Dh = D // H
```

Learn:

```python
W_qkv: Linear(D, 3 * D)
W_o:   Linear(D, D)
```

Or separate matrices if clearer:

```python
W_q: Linear(D, D)
W_k: Linear(D, D)
W_v: Linear(D, D)
W_o: Linear(D, D)
```

Default to fused QKV for performance, but keep code readable.

Shapes:

```python
x:   [B, N, D]
qkv: [B, N, 3D]
q:   [B, H, N, Dh]
k:   [B, H, N, Dh]
v:   [B, H, N, Dh]
```

Attention:

```python
scores = q @ k.transpose(-2, -1) / sqrt(Dh)
attn = softmax(scores, dim=-1)
out = attn @ v
out = rearrange(out, "b h n d -> b n (h d)")
out = W_o(out)
```

If `pos_embed_type == "rope_3d"`, apply RoPE to q and k before scores.

### MLP

Use:

```python
Linear(D, mlp_ratio * D)
GELU
Dropout
Linear(mlp_ratio * D, D)
Dropout
```

Default:

```python
mlp_ratio = 4
```

### Normalization

Use:

```python
nn.LayerNorm(embed_dim, eps=1e-6)
```

### Initialization

Use Transformer-friendly initialization:

* truncated normal for Linear weights,
* zeros for biases,
* LayerNorm weight = 1,
* LayerNorm bias = 0.

---

## VideoTransformerEncoder

Implement:

```python
class VideoTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_frames: int,
        tubelet_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        pos_embed_type: str,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        use_cls_token: bool = False,
    ):
        ...
```

For JEPA pretraining, default:

```python
use_cls_token = False
```

Because we need dense spatio-temporal tokens, not just a global representation.

Forward:

```python
def forward(
    self,
    video: torch.Tensor | None = None,
    tokens: torch.Tensor | None = None,
    visible_indices: torch.Tensor | None = None,
    return_all_tokens: bool = True,
) -> EncoderOutput:
    ...
```

Simpler initial implementation:

1. `patch_embed(video)` gives `[B, N, D]`.
2. Add absolute/sinusoidal position if relevant.
3. If `visible_indices` is provided, gather visible tokens.
4. Pass through Transformer blocks.
5. Return tokens and metadata.

Create:

```python
@dataclass
class EncoderOutput:
    tokens: torch.Tensor          # [B, N_visible, D]
    grid_shape: tuple[int, int, int]
    coords: torch.Tensor          # [N_visible, 3]
```

Important:

When using RoPE, if visible tokens are gathered, pass their corresponding coordinates to attention.

---

## Masking strategy

Implement multi-block spatio-temporal masking.

The mask generator operates on token grid:

```python
grid_shape = (T_tokens, H_tokens, W_tokens)
```

It should output:

```python
context_indices: [B, N_context]
target_indices:  [B, N_target]
```

Indices refer to flattened token positions.

Flattening order:

```python
index = t * H_tokens * W_tokens + y * W_tokens + x
```

### Masking objective

The context encoder sees context tokens.

The target encoder provides target embeddings for target tokens.

The predictor predicts embeddings for target token positions.

By construction the context is the **complement of the union of the target
blocks** — they never overlap. There is no independent "context ratio": you
sample target blocks, then context = all remaining tokens (optionally capped by
`max_keep`). This is what makes the task non-trivial; do not parameterize them
as two independent ratios.

### Batch-shared masks (critical)

Target blocks have random sizes, so a per-sample mask would give a *different*
number of context/target tokens per sample — which cannot be stacked into
`[B, N_context]` / `[B, N_target]` tensors. Follow the V-JEPA mask collator:
**sample one mask layout and apply the same mask to every clip in the batch.**
Then `N_context` and `N_target` are identical across the batch and stacking is
trivial. (To vary masks within a step, sample a few mask layouts and split the
batch into equal groups, one layout per group — never per-sample.)

### Multi-block strategy (V-JEPA / V-JEPA 2 defaults)

The SOTA recipe combines **two families of spatial blocks, each extended over
the full temporal extent** (`temporal_scale = 1.0`), removing up to ~90% of
tubelets:

- **short-range**: many small blocks (`num_blocks: 8`, `spatial_scale: 0.15`),
- **long-range**: a few large blocks (`num_blocks: 2`, `spatial_scale: 0.7`).

The masked region (union of all blocks) becomes the prediction target; the
visible complement is the context.

### Requirements

Implement:

```python
class MultiBlockMaskCollator:
    def __init__(
        self,
        grid_shape: tuple[int, int, int],   # (T_tokens, H_tokens, W_tokens)
        block_specs: list[BlockSpec],       # one entry per mask family below
        max_keep: int | None = None,        # optional cap on visible/context tokens
    ):
        ...

    def __call__(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Returns (context_indices, target_indices), each [B, N] with N
        identical across the batch (same mask reused for every sample)."""

@dataclass
class BlockSpec:
    num_blocks: int
    spatial_scale: tuple[float, float]      # fraction of H_tokens*W_tokens per block
    temporal_scale: tuple[float, float]     # 1.0 = block spans all frames
    aspect_ratio: tuple[float, float]       # block H:W aspect range
```

Default masks (from `facebookresearch/jepa` ViT-L/16):

```yaml
masking:
  - num_blocks: 8          # short-range
    spatial_scale: [0.15, 0.15]
    temporal_scale: [1.0, 1.0]
    aspect_ratio: [0.75, 1.5]
  - num_blocks: 2          # long-range
    spatial_scale: [0.7, 0.7]
    temporal_scale: [1.0, 1.0]
    aspect_ratio: [0.75, 1.5]
  max_keep: null
```

### Important

Avoid target blocks that are too tiny: prediction should require semantic
understanding, not local interpolation. The full-temporal-extent blocks above
force the model to reason across time, which is the point of a *video* JEPA.

**Do not use causal / future-frame masking in this repo.** It is tempting since
the downstream goal is a world model, but the encoder stage uses full-temporal
multiblock masking (`temporal_scale: 1.0`) — that is exactly the V-JEPA 2 recipe
whose frozen encoder powers the robotics AC stage. Predicting masked spatial
regions *across all frames* already forces motion understanding. Causal/dynamics
masking belongs to the action-conditioned predictor in the later stage, not
here. Keep `temporal_scale` configurable, but default and ship at `1.0`.

---

## JEPA model

Implement:

```python
class VideoJEPAModel(nn.Module):
    def __init__(
        self,
        context_encoder: VideoTransformerEncoder,
        target_encoder: VideoTransformerEncoder,
        predictor: LatentPredictor,
        loss_fn: nn.Module,
        ema_momentum: float,
    ):
        ...
```

### Forward pass

Input:

```python
video: [B, C, T, H, W]
context_indices: [B, N_context]
target_indices: [B, N_target]
```

Pseudo-code:

```python
# 1. Context path — context_encoder owns its patch_embed.
z_context = context_encoder(
    video=video,
    visible_indices=context_indices,
)  # EncoderOutput(tokens=[B, N_context, D], coords=[N_context, 3], ...)

# 2. Target path — target_encoder is a full EMA copy (its OWN patch_embed too).
with torch.no_grad():
    z_target = target_encoder(
        video=video,
        visible_indices=target_indices,
    )  # tokens=[B, N_target, D]

# 3. Predictor
z_pred = predictor(
    context_tokens=z_context.tokens,
    context_coords=z_context.coords,
    target_coords=z_target.coords,
)

# 4. Loss (target normalized + stop-gradient, see Loss section)
loss = loss_fn(z_pred, z_target.tokens.detach())
```

**Do not share a single `patch_embed` between the two encoders.** The target
encoder is an exponential-moving-average copy of the *entire* context encoder —
**including its patch embedding** — so its tokens must come from the slow EMA
weights, not from the context encoder's live weights. Computing `patch_embed`
once and feeding both paths breaks the EMA semantics. Each encoder embeds the
video itself; the patch-embed compute is cheap relative to the transformer
stacks, so there is nothing to optimize away here.

---

## Latent predictor

The predictor receives:

```python
z_context: [B, N_context, D]
target_coords: [B, N_target, 3] or [N_target, 3]
```

It outputs:

```python
z_pred: [B, N_target, D]
```

Architecture:

1. Project context tokens from encoder dim to predictor dim.
2. Create learned mask/query tokens for target positions.
3. Add or apply positional encodings to both context and target query tokens.
4. Concatenate:

```text
[context latent tokens] + [target query tokens]
```

5. Run through predictor Transformer.
6. Return only the target query outputs.
7. Project predictor dim back to encoder dim.

### Predictor config

```yaml
predictor:
  embed_dim: 512
  depth: 6
  num_heads: 8
  mlp_ratio: 4.0
  pos_embed_type: rope_3d
```

### Predictor forward

```python
def forward(
    self,
    context_tokens: torch.Tensor,
    context_coords: torch.Tensor,
    target_coords: torch.Tensor,
) -> torch.Tensor:
    ...
```

Output:

```python
[B, N_target, encoder_embed_dim]
```

---

## Loss functions

Implement at least:

```python
class LatentL1Loss(nn.Module)        # SOTA default (V-JEPA loss_exp = 1.0)
class LatentMSELoss(nn.Module)
class LatentSmoothL1Loss(nn.Module)
class LatentCosineLoss(nn.Module)
```

Default (matches V-JEPA / V-JEPA 2):

```yaml
loss:
  name: l1
  target_norm: layernorm   # LayerNorm (no affine) over feature dim of targets
```

Recommended default — **L1 on LayerNorm-normalized targets**, NOT L2
(`F.normalize`) normalization:

```python
# Normalize each target token over its feature dim. No learnable affine.
z_target = F.layer_norm(z_target, (z_target.shape[-1],))
loss = F.l1_loss(z_pred, z_target.detach())
```

Why LayerNorm and not `F.normalize`: V-JEPA normalizes targets with a
parameter-free LayerNorm (centering + per-feature scaling), which spreads
information across dimensions and discourages dimensional collapse. Projecting
onto the unit sphere with `F.normalize` (≈ cosine) is a *different*, weaker
objective — keep `LatentCosineLoss` available for ablation, but it is not the
default.

The optional variance/covariance regularizer (`reg_coeff`) from V-JEPA is **off
by default** (`0.0`); the EMA target + stop-gradient already prevent collapse.

The loss compares:

```text
predicted latent target tokens
vs
stop-gradient target encoder latent tokens
```

Never compare predicted pixels to real pixels.

There is no decoder.

There is no image reconstruction.

---

## EMA target encoder

Implement:

```python
@torch.no_grad()
def update_ema(context_encoder, target_encoder, momentum):
    for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_target.data.mul_(momentum).add_(p_context.data, alpha=1.0 - momentum)
```

At initialization:

```python
target_encoder.load_state_dict(context_encoder.state_dict())
requires_grad(target_encoder, False)
```

Only the context encoder and predictor receive gradients.

Target encoder is updated after each optimizer step.

Use schedule (V-JEPA ViT-L/16 default):

```python
ema_momentum_start = 0.998
ema_momentum_end = 1.0
```

Linear ramp over training (momentum increases toward 1.0, so the teacher freezes
progressively).

> SOTA note: V-JEPA 2 *simplified* this to a **fixed** EMA momentum (and fixed
> weight decay) instead of a ramp, and reports it works as well at scale.
> Support both: `ema.schedule: linear | constant`. Start with the ramp above for
> small runs.

---

## Training loop

Implement a custom trainer, not a huge framework dependency.

Use:

* PyTorch.
* torch.cuda.amp or torch.amp.
* optional distributed support later.
* gradient clipping.
* checkpointing.

### Precision and the GradScaler

`GradScaler` is **only for fp16**. In bf16 (the recommended default) the dynamic
range is wide enough that no loss scaling is needed — instantiate the scaler
disabled so the same code path works for both:

```python
use_fp16 = (precision == "fp16")
scaler = torch.amp.GradScaler(enabled=use_fp16)
amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
```

Training step:

```python
video = batch["video"]  # [B, C, T, H, W]

context_indices, target_indices = mask_collator(batch_size=B)

with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(precision != "fp32")):
    outputs = model(
        video=video,
        context_indices=context_indices,
        target_indices=target_indices,
    )
    loss = outputs.loss

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()         # no-op scaling when disabled (bf16/fp32)

if grad_clip_norm is not None:
    scaler.unscale_(optimizer)        # safe no-op when disabled
    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)

scaler.step(optimizer)
scaler.update()

ema_momentum = ema_schedule(step)
update_ema(context_encoder, target_encoder, ema_momentum)

scheduler.step()
```

Log:

```text
train/loss
train/lr
train/ema_momentum
train/grad_norm
train/mask_context_ratio
train/mask_target_ratio
train/z_pred_norm
train/z_target_norm
train/pred_target_cosine
```

---

## Optimizer

Use AdamW.

Default (V-JEPA ViT-L/16; lr is the peak after warmup):

```yaml
optimizer:
  name: adamw
  lr: 6.25e-4
  start_lr: 2.0e-4
  final_lr: 1.0e-6
  weight_decay: 0.04
  final_weight_decay: 0.4   # weight decay is ramped UP over training
  betas: [0.9, 0.95]
```

V-JEPA ramps weight decay from `weight_decay` → `final_weight_decay` on a cosine
schedule alongside the LR. V-JEPA 2 keeps it fixed; support both via
`final_weight_decay: null` (constant).

No weight decay on:

* biases,
* LayerNorm weights,
* positional embeddings,
* mask tokens.

Implement parameter groups.

---

## Scheduler

Implement both schedules and select by name:

* `cosine`: linear warmup → cosine decay (V-JEPA 1).
* `warmup_constant_decay`: linear warmup → constant plateau → linear cooldown
  (the **trapezoidal** schedule V-JEPA 2 switched to; decouples LR from total
  length and makes runs easy to extend).

Default:

```yaml
scheduler:
  name: warmup_constant_decay
  warmup_steps: 5000
  cooldown_steps: 5000
  max_steps: 100000
  min_lr: 1.0e-6
```

V-JEPA 2 reference proportions: warmup 12k / constant 228k / cooldown 12k
(≈5% warmup, 5% cooldown). Scale to your `max_steps`.

---

## Config system

Use YAML configs plus dataclasses.

Implement:

```python
TrainingConfig
ModelConfig
DataConfig
MaskingConfig
OptimizerConfig
SchedulerConfig
```

Use OmegaConf or plain PyYAML.

Prefer minimal dependencies.

Example `configs/tiny.yaml`:

```yaml
seed: 42
device: cuda
precision: bf16

data:
  root: ./data/videos
  clip_frames: 16
  frame_stride: 2
  image_size: 128
  batch_size: 2
  num_workers: 4

model:
  in_channels: 3
  image_size: 128
  num_frames: 16
  tubelet_size: [2, 16, 16]
  embed_dim: 384
  depth: 6
  num_heads: 6
  mlp_ratio: 4.0
  pos_embed_type: rope_3d
  drop_rate: 0.0
  attn_drop_rate: 0.0
  drop_path_rate: 0.1
  use_cls_token: false

predictor:
  embed_dim: 256
  depth: 4
  num_heads: 4
  mlp_ratio: 4.0
  pos_embed_type: rope_3d

masking:
  # Same mask reused across the whole batch (see Masking strategy).
  blocks:
    - num_blocks: 8          # short-range
      spatial_scale: [0.15, 0.15]
      temporal_scale: [1.0, 1.0]
      aspect_ratio: [0.75, 1.5]
    - num_blocks: 2          # long-range
      spatial_scale: [0.7, 0.7]
      temporal_scale: [1.0, 1.0]
      aspect_ratio: [0.75, 1.5]
  max_keep: null

loss:
  name: l1
  target_norm: layernorm
  reg_coeff: 0.0

optimizer:
  name: adamw
  lr: 0.000625
  start_lr: 0.0002
  final_lr: 0.000001
  weight_decay: 0.04
  final_weight_decay: 0.4
  betas: [0.9, 0.95]

scheduler:
  name: warmup_constant_decay
  warmup_steps: 1000
  cooldown_steps: 1000
  max_steps: 20000
  min_lr: 0.000001

ema:
  schedule: linear
  momentum_start: 0.998
  momentum_end: 1.0

training:
  max_steps: 20000
  log_every: 20
  save_every: 1000
  grad_clip_norm: 1.0
  output_dir: ./runs/tiny
```

---

## Dataset

Implement a simple video folder dataset.

Expected structure:

```text
data/videos/
├── video_001.mp4
├── video_002.mp4
└── video_003.mp4
```

Dataset behavior:

1. Randomly select a video.
2. Randomly sample a clip of `clip_frames`.
3. Apply temporal stride.
4. Decode frames.
5. Resize/crop.
6. Normalize.
7. Return tensor `[C, T, H, W]`.

Create abstraction:

```python
class VideoDataset(Dataset):
    ...
```

Use `torchvision.io` or `decord` if available.

Make backend configurable:

```yaml
data:
  backend: torchvision
```

Initial implementation can use torchvision.

Also provide a synthetic dataset fallback for tests:

```python
class SyntheticVideoDataset(Dataset):
    ...
```

This should generate moving colored squares or random tensors so the training loop can be tested without real videos.

---

## Transforms

Implement:

* resize shorter side,
* random resized crop,
* horizontal flip,
* color jitter optional,
* normalization.

Keep transforms video-consistent.

Important:

The same crop and flip must be applied to all frames of a clip.

Do not apply independent random crops per frame.

---

## Checkpointing

Save:

```python
{
    "step": step,
    "context_encoder": context_encoder.state_dict(),
    "target_encoder": target_encoder.state_dict(),
    "predictor": predictor.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "scaler": scaler.state_dict(),
    "config": config_dict,
}
```

Also provide an encoder-only export — this is the **deliverable** of this repo
and the input to the world-model stage, so it must be self-contained. Export the
**target (EMA) encoder**, which is the stable, better-quality encoder in JEPA:

```python
{
    "encoder": target_encoder.state_dict(),   # EMA weights, not context_encoder
    "model_config": model_config,             # embed_dim, depth, heads, tubelet, etc.
    "pos_embed_type": "rope_3d",              # + everything needed to rebuild coords
    "preprocessing": {                        # so the AC stage embeds clips identically
        "image_size": 224,
        "num_frames": 16,
        "tubelet_size": [2, 16, 16],
        "sampling_rate": 4,
        "normalization": {"mean": ..., "std": ...},
    },
}
```

A consumer must be able to: load this file, rebuild the encoder, call
`encoder.embed(video)` and get `tokens [B, N, D] + grid_shape + coords` with no
access to this training code. Verify that round-trip in a test.

Script:

```bash
python scripts/train.py --config configs/tiny.yaml
```

Resume:

```bash
python scripts/train.py --config configs/tiny.yaml --resume runs/tiny/checkpoints/latest.pt
```

Export:

```bash
python scripts/export_encoder.py --checkpoint runs/tiny/checkpoints/latest.pt --out encoder.pt
```

If you create `export_encoder.py`, add it to the file tree.

---

## Evaluation (downstream)

A falling pretraining loss does **not** prove the encoder is good — a partially
collapsed model can still drive L1 down. The only reliable signal is downstream
probing of the **frozen** encoder, exactly as V-JEPA evaluates.

Implement a frozen-encoder probe (V-JEPA uses an **attentive probe**: a small
cross-attention pooling head + linear classifier; a plain linear probe on
mean-pooled tokens is an acceptable simpler start):

```bash
python scripts/eval_probe.py --encoder encoder.pt --dataset <labeled_clips> --probe attentive
```

- Freeze the (target) encoder, train only the probe head.
- Report top-1 accuracy on a held-out split.
- Reference benchmarks in the literature: Something-Something v2 (motion),
  Kinetics-400 (appearance), Diving-48.

Cheap collapse monitors to log *during* pretraining (already in the training
metrics): if `train/z_target_norm` or the per-dimension target variance trends
toward zero, or `train/pred_target_cosine` saturates at 1.0 immediately, the
representation is collapsing — raise EMA momentum or check the stop-gradient.

---

## Tests

Write tests before or alongside implementation.

Minimum tests:

### test_patch_embed.py

Verify:

```python
video = torch.randn(2, 3, 16, 224, 224)
patch = VideoPatchEmbed(...)
tokens, grid = patch(video)
assert tokens.shape == (2, 1568, D)
assert grid == (8, 14, 14)
```

### test_rope.py

Verify:

* shape preservation,
* no NaNs,
* deterministic output,
* different positions produce different rotations.

### test_mask_generator.py

Verify:

* indices in valid range,
* no duplicates unless allowed,
* context and target shapes correct,
* batch support.

### test_encoder_shapes.py

Verify:

* encoder output shape,
* variable image size if supported,
* visible index gathering.

### test_jepa_forward.py

Verify:

```python
loss is scalar
z_pred.shape == z_target.shape
target encoder has no grad
context encoder has grad
predictor has grad
```

### test_ema.py

Verify target parameters move toward context parameters.

---

## Coding standards

Use:

```python
from __future__ import annotations
```

Use type hints.

Use dataclasses for structured outputs.

Avoid global state.

Avoid hardcoded shapes.

Every major module should have docstrings with expected tensor shapes.

Example:

```python
def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        x: Tensor of shape [B, N, D].
        coords: Optional integer coordinates of shape [N, 3] or [B, N, 3].

    Returns:
        Tensor of shape [B, N, D].
    """
```

---

## Important implementation details

### Index gathering

Implement a helper:

```python
def batched_gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    tokens: [B, N, D]
    indices: [B, M]
    returns: [B, M, D]
    """
```

Similarly for coordinates.

### Coordinate generation

Given grid:

```python
Tg, Hg, Wg = grid_shape
```

Generate:

```python
coords = [
    [t, y, x]
    for t in range(Tg)
    for y in range(Hg)
    for x in range(Wg)
]
```

Shape:

```python
[N, 3]
```

Flatten order must match patch flattening.

### Patch flattening order

If Conv3d gives:

```python
[B, D, Tg, Hg, Wg]
```

Flatten with:

```python
x = x.flatten(2).transpose(1, 2)
```

This produces order consistent with:

```text
t major, then y, then x
```

### Mixed precision

Support:

```yaml
precision: fp32 | fp16 | bf16
```

Prefer bf16 when available.

### Compile

Optionally support:

```yaml
training:
  compile: false
```

If true:

```python
model = torch.compile(model)
```

but keep disabled by default.

---

## Final mental model

The encoder is trained like this:

```text
video clip
↓
tubelet tokens
↓
split into context tokens and target tokens
↓
context encoder sees context
↓
target encoder EMA sees target
↓
predictor tries to predict target latents from context latents
↓
loss compares predicted target latents against stop-gradient target latents
```

Mathematically:

```text
z_context = E_theta(x_context)

with no grad:
    z_target = E_phi(x_target)

z_pred = P_psi(z_context, target_positions)

loss = distance(z_pred, stopgrad(z_target))

theta, psi updated by backprop
phi updated by EMA:
    phi = m * phi + (1 - m) * theta
```


[1]: https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/?utm_source=chatgpt.com "V-JEPA: The next step toward advanced machine intelligence"
[2]: https://github.com/facebookresearch/vjepa2?utm_source=chatgpt.com "PyTorch code and models for VJEPA2 self-supervised ..."
[3]: https://arxiv.org/html/2502.11664v4?utm_source=chatgpt.com "VRoPE: Rotary Position Embedding for Video Large ..."
