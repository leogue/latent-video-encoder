"""Latent predictor — a narrow transformer trained only during pretraining.

Predicts target-position latents from context latents. It is a throwaway
pretraining crutch (discarded after training); the action-conditioned world-model
predictor is a separate later stage.
"""

from __future__ import annotations

import torch
from torch import nn

from .pos_embed import RotaryEmbedding3D, sinusoidal_3d
from .transformer import Block, init_transformer_weights


class LatentPredictor(nn.Module):
    def __init__(
        self,
        encoder_embed_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        pos_embed_type: str = "rope_3d",
    ) -> None:
        super().__init__()
        assert pos_embed_type in ("rope_3d", "sinusoidal_3d", "none")
        self.pos_embed_type = pos_embed_type
        self.embed_dim = embed_dim

        self.ctx_proj = nn.Linear(encoder_embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.rope = RotaryEmbedding3D(embed_dim // num_heads) if pos_embed_type == "rope_3d" else None
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, 0.0, 0.0, 0.0, self.rope) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.out_proj = nn.Linear(embed_dim, encoder_embed_dim)
        self.apply(init_transformer_weights)
        # mask_token gets no weight decay; re-init after apply() (apply skips Parameters anyway).

    def _maybe_add_sincos(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        if self.pos_embed_type == "sinusoidal_3d":
            # coords: [B, N, 3]; build per-batch (same grid) -> use first row.
            pe = sinusoidal_3d(coords[0], self.embed_dim).unsqueeze(0)
            return x + pe
        return x

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_coords: torch.Tensor,
        target_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Predict latents at the target positions.

        Args:
            context_tokens: ``[B, N_ctx, D_enc]`` (context encoder output).
            context_coords: ``[B, N_ctx, 3]``.
            target_coords: ``[B, N_tgt, 3]``.

        Returns:
            ``[B, N_tgt, D_enc]`` predicted target latents.
        """
        b, n_tgt = target_coords.shape[0], target_coords.shape[1]
        ctx = self.ctx_proj(context_tokens)  # [B, N_ctx, D_pred]
        queries = self.mask_token.expand(b, n_tgt, -1)  # [B, N_tgt, D_pred]

        x = torch.cat((ctx, queries), dim=1)  # [B, N_ctx + N_tgt, D_pred]
        coords = torch.cat((context_coords, target_coords), dim=1)  # [B, N, 3]
        x = self._maybe_add_sincos(x, coords)

        for blk in self.blocks:
            x = blk(x, coords)
        x = self.norm(x)

        target_out = x[:, -n_tgt:]  # [B, N_tgt, D_pred]
        return self.out_proj(target_out)  # [B, N_tgt, D_enc]
