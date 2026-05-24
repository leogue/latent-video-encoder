from __future__ import annotations

import copy

import torch

from lve.config import Config, ModelConfig, build_encoder
from lve.ema import EMAMomentumSchedule, init_target_encoder, update_ema


def _encoder():
    mc = ModelConfig(image_size=64, num_frames=8, tubelet_size=(2, 16, 16),
                     embed_dim=96, depth=2, num_heads=6, drop_path_rate=0.0)
    return build_encoder(mc)


def test_init_copies_weights_and_freezes():
    ctx = _encoder()
    tgt = _encoder()
    init_target_encoder(ctx, tgt)
    for pc, pt in zip(ctx.parameters(), tgt.parameters()):
        assert torch.equal(pc, pt)
        assert not pt.requires_grad


def test_ema_moves_target_toward_context():
    ctx = _encoder()
    tgt = _encoder()
    init_target_encoder(ctx, tgt)
    # Perturb context so it differs from target.
    with torch.no_grad():
        for p in ctx.parameters():
            p.add_(torch.randn_like(p))

    before = copy.deepcopy([p.clone() for p in tgt.parameters()])
    update_ema(ctx, tgt, momentum=0.9)

    for p_before, p_ctx, p_tgt in zip(before, ctx.parameters(), tgt.parameters()):
        expected = 0.9 * p_before + 0.1 * p_ctx
        assert torch.allclose(p_tgt, expected, atol=1e-6)
        # Target moved closer to context than it was.
        assert (p_tgt - p_ctx).abs().sum() < (p_before - p_ctx).abs().sum()


def test_momentum_schedule():
    sched = EMAMomentumSchedule(0.998, 1.0, max_steps=100, schedule="linear")
    assert abs(sched(0) - 0.998) < 1e-9
    assert abs(sched(100) - 1.0) < 1e-9
    assert sched(50) > sched(0)
    const = EMAMomentumSchedule(0.998, 1.0, max_steps=100, schedule="constant")
    assert const(50) == 0.998
