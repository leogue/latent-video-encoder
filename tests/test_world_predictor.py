from __future__ import annotations

import torch

from lwp.config import WorldModelConfig, build_world_model
from lwp.dataset import SyntheticTrajectoryDataset
from lwp.predictor import ActionConditionedPredictor


def _predictor(hs=16, tg=8, d=32):
    return ActionConditionedPredictor(
        encoder_dim=d, state_dim=2, action_dim=2, num_spatial=hs,
        embed_dim=48, depth=2, num_heads=4, max_steps=tg,
    )


def test_predictor_output_shape():
    p = _predictor()
    b, tg, hs, d = 2, 8, 16, 32
    z = torch.randn(b, tg, hs, d)
    state = torch.randn(b, tg, 2)
    action = torch.randn(b, tg, 2)
    out = p(z, state, action)
    assert out.shape == (b, tg, hs, d)


def test_block_causal_no_future_leak():
    # Prediction at step t must not depend on inputs at steps > t.
    torch.manual_seed(0)
    p = _predictor().eval()
    b, tg, hs, d = 1, 8, 16, 32
    z = torch.randn(b, tg, hs, d)
    state = torch.randn(b, tg, 2)
    action = torch.randn(b, tg, 2)
    with torch.no_grad():
        out1 = p(z, state, action)
        # Perturb everything at the LAST step.
        z2, s2, a2 = z.clone(), state.clone(), action.clone()
        z2[:, -1] += 5.0
        s2[:, -1] += 5.0
        a2[:, -1] += 5.0
        out2 = p(z2, s2, a2)
    # All predictions before the last step are unchanged.
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)
    # The last step's prediction did change.
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_world_model_loss_and_frozen_encoder():
    cfg = WorldModelConfig()
    cfg.encoder.image_size = 64
    cfg.encoder.embed_dim = 96
    cfg.encoder.depth = 2
    cfg.predictor.embed_dim = 64
    cfg.predictor.depth = 2
    cfg.predictor.num_heads = 4
    cfg.world.rollout_steps = 3
    model = build_world_model(cfg)

    video = torch.randn(2, 3, 16, 64, 64)
    state = torch.randn(2, 8, 2)
    action = torch.randn(2, 8, 2)
    out = model(video, state, action)
    assert out.loss.dim() == 0
    assert out.loss_tf.dim() == 0 and out.loss_rollout.dim() == 0

    out.loss.backward()
    # Encoder frozen: no grads, requires_grad False.
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    # Predictor trained.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())


def test_rollout_shape():
    cfg = WorldModelConfig()
    cfg.encoder.image_size = 64
    cfg.encoder.embed_dim = 96
    cfg.encoder.depth = 2
    cfg.predictor.embed_dim = 64
    cfg.predictor.depth = 2
    cfg.predictor.num_heads = 4
    model = build_world_model(cfg)
    z = model.encode(torch.randn(2, 3, 16, 64, 64))
    state = torch.randn(2, 8, 2)
    action = torch.randn(2, 8, 2)
    preds = model.rollout(z[:, :1], state, action, steps=5)
    assert preds.shape == (2, 5, model.num_spatial, model.encoder.embed_dim)


def test_action_sensitivity_wired():
    # An untrained model is already action-sensitive (actions are wired in),
    # so the diagnostic must report a strictly positive value.
    torch.manual_seed(0)
    cfg = WorldModelConfig()
    cfg.encoder.image_size = 64
    cfg.encoder.embed_dim = 96
    cfg.encoder.depth = 2
    cfg.predictor.embed_dim = 64
    cfg.predictor.depth = 2
    cfg.predictor.num_heads = 4
    model = build_world_model(cfg)
    sens = model.action_sensitivity(
        torch.randn(2, 3, 16, 64, 64), torch.randn(2, 8, 2), torch.randn(2, 8, 2)
    )
    assert sens > 0.0


def test_synthetic_trajectory_dataset():
    ds = SyntheticTrajectoryDataset(length=4, num_frames=16, tubelet_t=2, image_size=64)
    s = ds[0]
    assert s["video"].shape == (3, 16, 64, 64)
    assert s["state"].shape == (8, 2)
    assert s["action"].shape == (8, 2)
    # Dynamics: pos_{t+1} = clamp(pos_t + action_t).
    expected = (s["state"][:-1] + s["action"][:-1]).clamp(0, 1)
    assert torch.allclose(s["state"][1:], expected, atol=1e-6)
