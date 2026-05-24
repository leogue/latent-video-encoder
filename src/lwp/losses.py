"""Latent dynamics losses.

The encoder is frozen, so targets are fixed encoder outputs (no EMA, no
stop-gradient games, no collapse risk) — this is plain supervised regression.
V-JEPA 2-AC uses an L1 reconstruction of encoder-derived latents.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between predicted and (detached) target latents."""
    return F.l1_loss(pred, target.detach())
