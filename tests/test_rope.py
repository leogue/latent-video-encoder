from __future__ import annotations

import torch

from lve.models import RotaryEmbedding3D
from lve.utils import build_coords


def _qkv(b, h, n, dh):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(b, h, n, dh, generator=g) for _ in range(3))


def test_rope_preserves_shape_and_no_nan():
    rope = RotaryEmbedding3D(head_dim=48)
    coords = build_coords((4, 4, 4))
    q, k, _ = _qkv(2, 6, 64, 48)
    qr, kr = rope.rotate_qk(q, k, coords)
    assert qr.shape == q.shape and kr.shape == k.shape
    assert torch.isfinite(qr).all() and torch.isfinite(kr).all()


def test_rope_does_not_touch_v():
    # V is never passed to rope.apply; rotating q,k must leave a separate v untouched.
    rope = RotaryEmbedding3D(head_dim=48)
    coords = build_coords((2, 4, 4))
    q, k, v = _qkv(1, 6, 32, 48)
    v_before = v.clone()
    rope.rotate_qk(q, k, coords)
    assert torch.equal(v, v_before)


def test_rope_deterministic():
    rope = RotaryEmbedding3D(head_dim=48)
    coords = build_coords((2, 4, 4))
    q, k, _ = _qkv(1, 6, 32, 48)
    q1, _ = rope.rotate_qk(q, k, coords)
    q2, _ = rope.rotate_qk(q, k, coords)
    assert torch.equal(q1, q2)


def test_rope_different_positions_differ():
    rope = RotaryEmbedding3D(head_dim=48)
    q = torch.ones(1, 1, 2, 48)
    k = torch.ones(1, 1, 2, 48)
    coords = torch.tensor([[0, 0, 0], [3, 2, 1]])
    qr, _ = rope.rotate_qk(q, k, coords)
    # Same input vector, different coords -> different rotated output.
    assert not torch.allclose(qr[0, 0, 0], qr[0, 0, 1])


def test_rope_variable_sequence_length():
    rope = RotaryEmbedding3D(head_dim=48)
    for n_grid in [(1, 2, 2), (4, 4, 4), (8, 4, 4)]:
        coords = build_coords(n_grid)
        n = coords.shape[0]
        q, k, _ = _qkv(1, 6, n, 48)
        qr, kr = rope.rotate_qk(q, k, coords)
        assert qr.shape == (1, 6, n, 48)
