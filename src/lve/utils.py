"""Shared tensor helpers: coordinate generation and batched gathering.

The flattening order used everywhere in this codebase is **t-major, then y,
then x**:

    index = t * (Hg * Wg) + y * Wg + x

This matches a Conv3d output ``[B, D, Tg, Hg, Wg]`` flattened with
``x.flatten(2).transpose(1, 2)``. Coordinate generation below must stay
consistent with it.
"""

from __future__ import annotations

import torch


def build_coords(grid_shape: tuple[int, int, int], device: torch.device | None = None) -> torch.Tensor:
    """Integer (t, y, x) coordinates for every token in the grid.

    Args:
        grid_shape: ``(Tg, Hg, Wg)``.
        device: optional target device.

    Returns:
        LongTensor of shape ``[N, 3]`` with ``N = Tg * Hg * Wg``, ordered
        t-major then y then x so row ``i`` corresponds to flattened index ``i``.
    """
    tg, hg, wg = grid_shape
    t = torch.arange(tg, device=device)
    y = torch.arange(hg, device=device)
    x = torch.arange(wg, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    coords = torch.stack([tt.reshape(-1), yy.reshape(-1), xx.reshape(-1)], dim=-1)
    return coords.long()


def batched_gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather tokens along the sequence dimension, per batch element.

    Args:
        tokens: ``[B, N, D]``.
        indices: ``[B, M]`` (long) of token positions in ``[0, N)``.

    Returns:
        ``[B, M, D]``.
    """
    b, _, d = tokens.shape
    m = indices.shape[1]
    idx = indices.unsqueeze(-1).expand(b, m, d)
    return torch.gather(tokens, dim=1, index=idx)


def gather_coords(coords: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather coordinates for a (possibly batched) set of indices.

    Args:
        coords: ``[N, 3]`` shared grid coordinates.
        indices: ``[B, M]`` or ``[M]`` token positions.

    Returns:
        ``[B, M, 3]`` if indices are batched, else ``[M, 3]``.
    """
    return coords[indices]
