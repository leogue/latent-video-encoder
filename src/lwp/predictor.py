"""Action-conditioned latent predictor (V-JEPA 2-AC style).

A block-causal Transformer over a robot trajectory. Each timestep contributes
three token types — visual (frozen-encoder patch features), proprioceptive
state, and action — projected by separate input heads. Block-causal attention
means a token at step ``t`` attends to all tokens at steps ``<= t``. From the
visual tokens at step ``t`` the model predicts the visual latent at step
``t+1``.

We use learned step / type / spatial position embeddings here (not RoPE): the
frozen-encoder tokens already carry spatial structure, and the predictor mainly
needs to know *which timestep* and *which token type* each token is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from lve.models.transformer import DropPath, Mlp, init_transformer_weights

# Token-type ids.
VISUAL, STATE, ACTION = 0, 1, 2


class MaskedAttention(nn.Module):
    """Multi-head self-attention with an explicit boolean attention mask."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """``x`` ``[B, L, D]``; ``attn_mask`` boolean ``[L, L]`` (True = may attend)."""
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out))


class PredictorBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MaskedAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ActionConditionedPredictor(nn.Module):
    """Predicts the next-step visual latent from past visual/state/action tokens."""

    def __init__(
        self,
        encoder_dim: int,
        state_dim: int,
        action_dim: int,
        num_spatial: int,  # Hs = Hg * Wg visual tokens per timestep
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        max_steps: int = 64,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_spatial = num_spatial
        self.block = num_spatial + 2  # visual tokens + state + action

        self.visual_in = nn.Linear(encoder_dim, embed_dim)
        self.state_in = nn.Linear(state_dim, embed_dim)
        self.action_in = nn.Linear(action_dim, embed_dim)

        self.step_embed = nn.Parameter(torch.zeros(1, max_steps, 1, embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, num_spatial, embed_dim))
        for p in (self.step_embed, self.type_embed, self.spatial_embed):
            nn.init.trunc_normal_(p, std=0.02)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [PredictorBlock(embed_dim, num_heads, mlp_ratio, dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.visual_out = nn.Linear(embed_dim, encoder_dim)
        self.apply(init_transformer_weights)

    def _assemble(
        self, z: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the chronological token sequence and its per-token step index.

        Args:
            z: visual latents ``[B, T, Hs, D_enc]``.
            state: ``[B, T, state_dim]``.
            action: ``[B, T, action_dim]``.

        Returns:
            ``seq`` ``[B, T*block, embed_dim]`` and ``step_idx`` ``[T*block]``.
        """
        b, t = z.shape[0], z.shape[1]
        step = self.step_embed[:, :t]  # [1, T, 1, E]

        vis = self.visual_in(z) + self.spatial_embed + step + self.type_embed[VISUAL]
        st = self.state_in(state).unsqueeze(2) + step + self.type_embed[STATE]  # [B,T,1,E]
        ac = self.action_in(action).unsqueeze(2) + step + self.type_embed[ACTION]

        block = torch.cat([vis, st, ac], dim=2)  # [B, T, block, E]
        seq = block.reshape(b, t * self.block, -1)
        step_idx = torch.arange(t, device=z.device).repeat_interleave(self.block)
        return seq, step_idx

    def forward(self, z: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict each step's *next* visual latent.

        Returns ``[B, T, Hs, D_enc]`` where output ``[:, t]`` predicts ``z[:, t+1]``
        (the last step's output has no ground-truth target).
        """
        b, t = z.shape[0], z.shape[1]
        seq, step_idx = self._assemble(z, state, action)
        attn_mask = step_idx.unsqueeze(0) <= step_idx.unsqueeze(1)  # [L, L] block-causal

        for blk in self.blocks:
            seq = blk(seq, attn_mask)
        seq = self.norm(seq)

        block = seq.reshape(b, t, self.block, -1)
        visual_hidden = block[:, :, : self.num_spatial]  # [B, T, Hs, E]
        return self.visual_out(visual_hidden)  # [B, T, Hs, D_enc]
