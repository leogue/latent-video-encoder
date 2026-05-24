# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Implemented. The package lives under `src/lve/` and the full V-JEPA encoder
pretraining pipeline runs end-to-end (model, masking, losses, EMA, data,
trainer, scripts, tests). `README.md` is the GitHub-facing presentation;
**[`docs/DESIGN.md`](docs/DESIGN.md) is the detailed design spec / source of
truth** for architecture rationale, defaults, and SOTA references; the code
follows it. When changing behavior, keep code and spec in sync.

Package layout: `models/` (patch_embed, pos_embed, transformer, encoder,
predictor, jepa), `masking.py`, `losses.py`, `ema.py`, `optim.py`, `config.py`
(dataclasses + YAML loader + factories), `trainer.py`, `data/` (synthetic +
torchvision folder dataset + clip-consistent transforms), `probe.py`. Scripts in
`scripts/`, configs in `configs/`, tests in `tests/`.

## Tooling

The project uses [uv](https://docs.astral.sh/uv/) (`.python-version` = 3.10,
`src/` layout, `pyproject.toml` with hatchling). Deps: `torch`, `pyyaml`,
`pytest` (dev). `pytest` is configured with `pythonpath = ["src"]`.

- `uv sync` — install the project (needed once / after dependency changes).
- `uv run pytest` — run all tests; `uv run pytest tests/test_rope.py::test_rope_deterministic` for one.
- `uv run python scripts/train.py --config configs/tiny.yaml` — train (synthetic data by default, CPU-friendly).
- `uv run python scripts/train.py --config configs/tiny.yaml --resume runs/tiny/checkpoints/latest.pt` — resume.
- `uv run python scripts/export_encoder.py --checkpoint <ckpt> --out encoder.pt` — export the frozen EMA encoder (the deliverable).
- `uv run python scripts/eval_probe.py --encoder encoder.pt --num-classes N` — attentive probe (needs a labeled dataset wired into `build_labeled_dataset`).

Note: the env has no `numpy`; torch prints a harmless "Failed to initialize
NumPy" warning. Don't add numpy just to silence it.

### Gotcha: do not name a method `apply` on an `nn.Module`

`RotaryEmbedding3D` exposes `rotate_qk(q, k, coords)`, not `apply(...)` —
`nn.Module.apply` is used for weight init and a custom `apply` shadows it,
breaking `self.apply(init_fn)`.

## Architecture (the big picture)

This is **V-JEPA**: latent-space joint-embedding prediction. There is **no
decoder and no pixel reconstruction** — loss is always computed between
predicted latents and stop-gradient target latents.

**Scope:** this repo trains the **encoder only**. The eventual goal is a world
model for robotics (action-conditioned predictor, V-JEPA 2-AC style) but that is
a separate later stage built on the *frozen* encoder shipped here. Do not add
action conditioning, a rollout/dynamics predictor, or a decoder. The JEPA
predictor here is a throwaway pretraining crutch. The deliverable is an
**encoder-only export of the EMA (target) encoder** (self-contained: weights +
model config + preprocessing), exposing `embed(video) -> tokens [B,N,D] +
grid_shape + coords`. See the "Scope and downstream contract" section in the spec.

Three neural components:
1. `context_encoder` (`VideoTransformerEncoder`) — sees only context tokens, receives gradients.
2. `target_encoder` (same architecture) — sees target tokens under `no_grad`, updated by **EMA**, never by backprop. Initialized from a copy of the context encoder's weights.
3. `predictor` (`LatentPredictor`) — predicts target-position latents from context latents + target coordinates.

Training step flow:
```
video [B,C,T,H,W]
  → tubelet patch embed (Conv3d) → tokens [B,N,D] + grid_shape (Tg,Hg,Wg) + coords [N,3]
  → mask generator splits flattened token indices into context_indices / target_indices
  → context path: gather context tokens → context_encoder → z_context
  → target path (no_grad): gather target tokens → target_encoder → z_target
  → predictor(z_context, context_coords, target_coords) → z_pred
  → loss(z_pred, stopgrad(z_target))
  → backprop updates context_encoder + predictor
  → EMA updates target_encoder
```

Key invariants that span multiple modules and are easy to get wrong:
- **Flattening order is `t`-major, then `y`, then `x`**: `index = t*Hg*Wg + y*Wg + x`. The Conv3d flatten (`x.flatten(2).transpose(1,2)`) and the `coords` generation must produce the *same* order. Mask indices reference these flattened positions.
- **RoPE (`rope_3d`) is applied to Q and K inside attention**, not added to token embeddings — and it must not modify V. When visible/target tokens are gathered, their corresponding `coords` must be passed through to attention so rotations use correct positions.
- Position encoding is pluggable: `learned_abs | sinusoidal_3d | rope_3d | none`.
- `use_cls_token = False` by default — JEPA needs dense spatio-temporal tokens, not a pooled global vector.

## Config system

YAML configs (e.g. `configs/tiny.yaml`, fully specified in the README) backed by
dataclasses: `TrainingConfig`, `ModelConfig`, `DataConfig`, `MaskingConfig`,
`OptimizerConfig`, `SchedulerConfig`. Prefer minimal deps (PyYAML or OmegaConf).

## Conventions from the spec

- Internal video tensor layout is `[B, C, T, H, W]`.
- Use `from __future__ import annotations`, type hints, and dataclasses for structured outputs (e.g. `EncoderOutput`).
- Docstrings on major modules must state expected tensor shapes.
- Optimizer is AdamW with **parameter groups**: no weight decay on biases, LayerNorm weights, positional embeddings, or mask tokens. Weight decay is ramped up over training (0.04 → 0.4).
- EMA momentum ramps linearly 0.998 → 1.0 (V-JEPA 2 alternative: fixed momentum).
- LR schedule defaults to **warmup → constant → cooldown** (trapezoidal, V-JEPA 2); cosine decay is also supported.
- Loss is **L1** between `z_pred` and `LayerNorm(z_target)` (parameter-free LayerNorm over the feature dim), *not* L2/`F.normalize`. The mask collator reuses **one mask across the whole batch** so `N_context`/`N_target` are fixed and stackable; context is the complement of the (disjoint) target blocks.
- Provide a `SyntheticVideoDataset` (moving squares / random tensors) so the training loop and tests run without real video files.
- Video transforms (crop, flip, jitter) must be applied **identically across all frames of a clip** — never independent per-frame randomness.

## Tests

The README enumerates the required test files and their assertions:
`test_patch_embed.py`, `test_rope.py`, `test_mask_generator.py`,
`test_encoder_shapes.py`, `test_jepa_forward.py`, `test_ema.py`. Write/maintain
these alongside implementation.
