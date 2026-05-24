"""Downstream evaluation: frozen-encoder attentive probe.

A falling pretraining loss does not prove the encoder is good. The reliable
signal is probing the *frozen* encoder. This provides an attentive probe (a
learned query cross-attends to the token sequence, then a linear classifier) and
a helper to rebuild an exported encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from .models import VideoTransformerEncoder


def load_exported_encoder(path: str) -> VideoTransformerEncoder:
    """Rebuild a frozen encoder from an export produced by export_encoder.py."""
    export = torch.load(path, map_location="cpu", weights_only=False)
    mc = export["model_config"]
    encoder = VideoTransformerEncoder(
        image_size=mc["image_size"],
        num_frames=mc["num_frames"],
        tubelet_size=tuple(mc["tubelet_size"]),
        in_channels=mc["in_channels"],
        embed_dim=mc["embed_dim"],
        depth=mc["depth"],
        num_heads=mc["num_heads"],
        mlp_ratio=mc["mlp_ratio"],
        pos_embed_type=mc["pos_embed_type"],
    )
    encoder.load_state_dict(export["encoder"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


class AttentiveProbe(nn.Module):
    """Cross-attention pooling head + linear classifier over frozen tokens."""

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args: ``tokens`` ``[B, N, D]`` (frozen encoder output). Returns logits ``[B, num_classes]``."""
        b = tokens.shape[0]
        q = self.query.expand(b, -1, -1)
        pooled, _ = self.attn(q, tokens, tokens)  # [B, 1, D]
        return self.head(self.norm(pooled.squeeze(1)))
