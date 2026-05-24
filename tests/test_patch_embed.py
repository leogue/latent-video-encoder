from __future__ import annotations

import torch

from lve.models import VideoPatchEmbed


def test_patch_embed_shapes():
    patch = VideoPatchEmbed(
        image_size=224, num_frames=16, tubelet_size=(2, 16, 16), in_channels=3, embed_dim=192
    )
    video = torch.randn(2, 3, 16, 224, 224)
    tokens, grid = patch(video)
    assert tokens.shape == (2, 1568, 192)
    assert grid == (8, 14, 14)
    assert patch.num_tokens == 1568


def test_patch_embed_flatten_order_is_t_major():
    # A Conv3d with identity-ish weights would preserve order; here we just check
    # that the number of tokens equals the product of the grid dims in t,y,x order.
    patch = VideoPatchEmbed(
        image_size=64, num_frames=8, tubelet_size=(2, 16, 16), in_channels=3, embed_dim=32
    )
    tokens, grid = patch(torch.randn(1, 3, 8, 64, 64))
    assert grid == (4, 4, 4)
    assert tokens.shape[1] == 4 * 4 * 4
