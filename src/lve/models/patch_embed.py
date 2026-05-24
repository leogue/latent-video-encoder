"""Tubelet patch embedding via Conv3d."""

from __future__ import annotations

import torch
from torch import nn


def _pair(x: int | tuple[int, int]) -> tuple[int, int]:
    return (x, x) if isinstance(x, int) else x


class VideoPatchEmbed(nn.Module):
    """Convert a video into spatio-temporal tubelet tokens.

    A ``Conv3d`` with kernel = stride = ``tubelet_size`` projects each
    non-overlapping tubelet to ``embed_dim``.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_frames: int,
        tubelet_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        h, w = _pair(image_size)
        pt, ph, pw = tubelet_size
        assert num_frames % pt == 0, f"num_frames {num_frames} not divisible by tubelet_t {pt}"
        assert h % ph == 0 and w % pw == 0, "image_size not divisible by tubelet spatial size"
        self.tubelet_size = tubelet_size
        self.grid_shape = (num_frames // pt, h // ph, w // pw)
        self.num_tokens = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=tubelet_size, stride=tubelet_size)

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Embed a video clip.

        Args:
            video: ``[B, C, T, H, W]``.

        Returns:
            ``tokens`` of shape ``[B, N, D]`` (t-major, then y, then x) and
            ``grid_shape = (Tg, Hg, Wg)``.
        """
        x = self.proj(video)  # [B, D, Tg, Hg, Wg]
        grid_shape = (x.shape[2], x.shape[3], x.shape[4])
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x, grid_shape
