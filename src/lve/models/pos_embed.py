"""Position encodings: 3D RoPE, factorized 3D sinusoidal, and learned absolute.

`rope_3d` is the default and is applied to Q/K *inside attention* (never to V,
never added to token embeddings). `sinusoidal_3d` and `learned_abs` are additive
embeddings added to tokens before the transformer blocks.
"""

from __future__ import annotations

import torch
from torch import nn


def _rotary_dims(head_dim: int) -> tuple[int, int, int]:
    """Split ``head_dim`` into three ~equal *even* segments (t, y, x).

    Any remainder is left unrotated. Returns ``(d_t, d_y, d_x)``.
    """
    base = head_dim // 3
    base -= base % 2  # rotation operates on pairs -> must be even
    return base, base, base


def _rotate_half_segments(x: torch.Tensor, dims: tuple[int, int, int]) -> torch.Tensor:
    """Apply the RoPE ``rotate_half`` operation independently per axis segment.

    Args:
        x: ``[..., sum(dims)]``.
        dims: per-axis rotated sizes.

    Returns:
        Same shape as ``x``.
    """
    out = []
    start = 0
    for d in dims:
        seg = x[..., start : start + d]
        half = d // 2
        x1, x2 = seg[..., :half], seg[..., half:]
        out.append(torch.cat((-x2, x1), dim=-1))
        start += d
    return torch.cat(out, dim=-1)


class RotaryEmbedding3D(nn.Module):
    """3D rotary position embedding applied to Q and K.

    The feature (head) dimension is partitioned into three ~equal segments for
    the temporal, height, and width axes; standard 1D RoPE is applied to each
    using that axis' integer coordinate.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.dims = _rotary_dims(head_dim)
        self.d_rot = sum(self.dims)

    def _cos_sin(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin caches for the given coordinates.

        Args:
            coords: ``[..., 3]`` integer (t, y, x) coordinates.

        Returns:
            ``cos`` and ``sin``, each ``[..., d_rot]``.
        """
        coords = coords.float()
        cos_list, sin_list = [], []
        for axis, d in enumerate(self.dims):
            half = d // 2
            inv_freq = 1.0 / (
                self.theta ** (torch.arange(half, device=coords.device).float() / half)
            )  # [half]
            ang = coords[..., axis : axis + 1] * inv_freq  # [..., half]
            emb = torch.cat((ang, ang), dim=-1)  # [..., d]
            cos_list.append(emb.cos())
            sin_list.append(emb.sin())
        return torch.cat(cos_list, dim=-1), torch.cat(sin_list, dim=-1)

    def rotate_qk(
        self, q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K according to their token coordinates.

        Args:
            q, k: ``[B, H, N, head_dim]``.
            coords: ``[B, N, 3]`` or ``[N, 3]`` integer coordinates.

        Returns:
            Rotated ``(q, k)`` with identical shapes to the inputs.
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(q.shape[0], -1, -1)
        cos, sin = self._cos_sin(coords)  # [B, N, d_rot]
        cos = cos.unsqueeze(1)  # broadcast over heads -> [B, 1, N, d_rot]
        sin = sin.unsqueeze(1)

        def rope(x: torch.Tensor) -> torch.Tensor:
            x_rot, x_pass = x[..., : self.d_rot], x[..., self.d_rot :]
            x_rot = x_rot * cos + _rotate_half_segments(x_rot, self.dims) * sin
            return torch.cat((x_rot, x_pass), dim=-1)

        return rope(q), rope(k)


def sinusoidal_3d(coords: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Factorized deterministic 3D sinusoidal embedding.

    ``dim`` is split across the three axes (``D_t + D_h + D_w = dim``, the
    temporal axis absorbs any remainder) and concatenated.

    Args:
        coords: ``[N, 3]`` integer coordinates.
        dim: total embedding dimension ``D``.

    Returns:
        ``[N, dim]`` float tensor (no gradient).
    """
    # Per-axis split, each even; remainder goes to the time axis.
    per = dim // 3
    per -= per % 2
    d_h = d_w = per
    d_t = dim - d_h - d_w
    parts = []
    for axis, d in zip((0, 1, 2), (d_t, d_h, d_w)):
        half = d // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(half, device=coords.device).float() / max(half, 1))
        )
        ang = coords[:, axis : axis + 1].float() * inv_freq  # [N, half]
        emb = torch.cat((ang.sin(), ang.cos()), dim=-1)  # [N, 2*half]
        if emb.shape[-1] < d:  # pad if d was odd
            emb = torch.cat((emb, emb.new_zeros(emb.shape[0], d - emb.shape[-1])), dim=-1)
        parts.append(emb)
    return torch.cat(parts, dim=-1)
