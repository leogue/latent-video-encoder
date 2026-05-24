"""Pre-norm Transformer block with optional 3D RoPE attention.

    x = x + DropPath(SelfAttention(LayerNorm(x)))
    x = x + DropPath(MLP(LayerNorm(x)))
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pos_embed import RotaryEmbedding3D


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero whole samples in the residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x / keep * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    """Multi-head self-attention with fused QKV and optional 3D RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: RotaryEmbedding3D | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        """Args: ``x`` ``[B, N, D]``, ``coords`` ``[B, N, 3]`` or ``[N, 3]`` (RoPE only).

        Returns: ``[B, N, D]``.
        """
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope is not None:
            assert coords is not None, "rope_3d requires coords"
            q, k = self.rope.rotate_qk(q, k, coords)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0
        )  # [B, H, N, Dh]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path_rate: float,
        rope: RotaryEmbedding3D | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, attn_drop, drop, rope)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), coords))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_transformer_weights(module: nn.Module) -> None:
    """Truncated-normal Linear weights, zero bias; LayerNorm weight 1 / bias 0."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
