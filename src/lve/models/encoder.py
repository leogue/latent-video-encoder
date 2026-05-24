"""VideoTransformerEncoder — the deliverable of this repo.

Produces dense spatio-temporal tokens (no CLS by default) consumed by the JEPA
predictor during pretraining and by the downstream world-model stage at
inference. See the "Scope and downstream contract" section of the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..utils import batched_gather_tokens, build_coords, gather_coords
from .patch_embed import VideoPatchEmbed
from .pos_embed import RotaryEmbedding3D, sinusoidal_3d
from .transformer import Block, init_transformer_weights


@dataclass
class EncoderOutput:
    tokens: torch.Tensor  # [B, N, D]
    grid_shape: tuple[int, int, int]
    coords: torch.Tensor  # [B, N, 3] for the (possibly gathered) tokens


class VideoTransformerEncoder(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        num_frames: int,
        tubelet_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pos_embed_type: str = "rope_3d",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()
        assert not use_cls_token, "JEPA needs dense tokens; CLS token unsupported"
        assert pos_embed_type in ("rope_3d", "sinusoidal_3d", "learned_abs", "none")
        self.pos_embed_type = pos_embed_type
        self.embed_dim = embed_dim

        self.patch_embed = VideoPatchEmbed(
            image_size, num_frames, tubelet_size, in_channels, embed_dim
        )
        n = self.patch_embed.num_tokens

        self.rope = RotaryEmbedding3D(embed_dim // num_heads) if pos_embed_type == "rope_3d" else None
        if pos_embed_type == "learned_abs":
            self.pos_embed = nn.Parameter(torch.zeros(1, n, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # Stochastic depth, linearly increasing with depth.
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate, dpr[i], self.rope)
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.apply(init_transformer_weights)

    def _add_pos(self, tokens: torch.Tensor, coords_full: torch.Tensor) -> torch.Tensor:
        if self.pos_embed_type == "learned_abs":
            return tokens + self.pos_embed
        if self.pos_embed_type == "sinusoidal_3d":
            return tokens + sinusoidal_3d(coords_full, self.embed_dim).unsqueeze(0)
        return tokens  # rope_3d / none: nothing added here

    def forward(
        self,
        video: torch.Tensor,
        visible_indices: torch.Tensor | None = None,
    ) -> EncoderOutput:
        """Encode a clip, optionally restricted to a subset of token positions.

        Args:
            video: ``[B, C, T, H, W]``.
            visible_indices: ``[B, M]`` token positions to keep, or ``None`` for
                the full grid.

        Returns:
            EncoderOutput with ``tokens`` ``[B, M, D]`` and ``coords`` ``[B, M, 3]``.
        """
        tokens, grid_shape = self.patch_embed(video)  # [B, N, D]
        coords_full = build_coords(grid_shape, device=tokens.device)  # [N, 3]
        tokens = self._add_pos(tokens, coords_full)

        if visible_indices is not None:
            tokens = batched_gather_tokens(tokens, visible_indices)  # [B, M, D]
            coords = gather_coords(coords_full, visible_indices)  # [B, M, 3]
        else:
            b = tokens.shape[0]
            coords = coords_full.unsqueeze(0).expand(b, -1, -1)

        for blk in self.blocks:
            tokens = blk(tokens, coords)
        tokens = self.norm(tokens)
        return EncoderOutput(tokens=tokens, grid_shape=grid_shape, coords=coords)

    @torch.no_grad()
    def embed(self, video: torch.Tensor) -> EncoderOutput:
        """Deterministic full-grid embedding for downstream/inference use."""
        was_training = self.training
        self.eval()
        out = self.forward(video, visible_indices=None)
        if was_training:
            self.train()
        return out
