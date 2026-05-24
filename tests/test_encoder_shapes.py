from __future__ import annotations

import pytest
import torch

from lve.models import VideoTransformerEncoder

KW = dict(
    num_frames=8,
    tubelet_size=(2, 16, 16),
    in_channels=3,
    embed_dim=96,
    depth=2,
    num_heads=6,
)


def _encoder(pos):
    return VideoTransformerEncoder(image_size=64, pos_embed_type=pos, **KW)


@pytest.mark.parametrize("pos", ["rope_3d", "sinusoidal_3d", "learned_abs", "none"])
def test_full_grid_output(pos):
    enc = _encoder(pos)
    video = torch.randn(2, 3, 8, 64, 64)
    out = enc(video)
    n = enc.patch_embed.num_tokens  # 4*4*4 = 64
    assert out.tokens.shape == (2, n, 96)
    assert out.grid_shape == (4, 4, 4)
    assert out.coords.shape == (2, n, 3)


def test_visible_index_gathering():
    enc = _encoder("rope_3d")
    video = torch.randn(2, 3, 8, 64, 64)
    visible = torch.randint(0, 64, (2, 10))
    out = enc(video, visible_indices=visible)
    assert out.tokens.shape == (2, 10, 96)
    assert out.coords.shape == (2, 10, 3)


def test_embed_is_deterministic_and_no_grad():
    enc = _encoder("rope_3d")
    video = torch.randn(1, 3, 8, 64, 64)
    o1 = enc.embed(video)
    o2 = enc.embed(video)
    assert torch.allclose(o1.tokens, o2.tokens)
    assert not o1.tokens.requires_grad


def test_no_cls_token_assertion():
    with pytest.raises(AssertionError):
        VideoTransformerEncoder(image_size=64, pos_embed_type="rope_3d", use_cls_token=True, **KW)
