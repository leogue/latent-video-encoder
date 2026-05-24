from __future__ import annotations

import torch

from lve.config import Config, DataConfig, ModelConfig, PredictorConfig, build_jepa
from lve.masking import BlockSpec, MultiBlockMaskCollator


def _small_cfg():
    cfg = Config()
    cfg.model = ModelConfig(
        image_size=64, num_frames=8, tubelet_size=(2, 16, 16), embed_dim=96, depth=2, num_heads=6
    )
    cfg.predictor = PredictorConfig(embed_dim=64, depth=2, num_heads=4)
    cfg.data = DataConfig(image_size=64, clip_frames=8)
    return cfg


def _collator(grid):
    specs = [BlockSpec(4, (0.15, 0.15), (1.0, 1.0), (0.75, 1.5)),
             BlockSpec(1, (0.5, 0.5), (1.0, 1.0), (0.75, 1.5))]
    return MultiBlockMaskCollator(grid, specs, generator=torch.Generator().manual_seed(0))


def test_jepa_forward_loss_and_shapes():
    cfg = _small_cfg()
    model = build_jepa(cfg)
    grid = model.context_encoder.patch_embed.grid_shape
    col = _collator(grid)
    ctx, tgt = col(batch_size=2)

    video = torch.randn(2, 3, 8, 64, 64)
    out = model(video, ctx, tgt)

    assert out.loss.dim() == 0  # scalar
    assert out.z_pred.shape == out.z_target.shape
    assert out.z_pred.shape == (2, tgt.shape[1], cfg.model.embed_dim)


def test_gradient_flow():
    cfg = _small_cfg()
    model = build_jepa(cfg)
    grid = model.context_encoder.patch_embed.grid_shape
    ctx, tgt = _collator(grid)(batch_size=2)
    out = model(torch.randn(2, 3, 8, 64, 64), ctx, tgt)
    out.loss.backward()

    # Context encoder + predictor receive gradients.
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.context_encoder.parameters())
    assert any(p.grad is not None for p in model.predictor.parameters())

    # Target encoder is frozen: no requires_grad, no grad.
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
