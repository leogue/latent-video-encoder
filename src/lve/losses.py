"""Latent prediction losses.

Default is L1 over LayerNorm-normalized targets (parameter-free LayerNorm over
the feature dim), matching V-JEPA. This is NOT L2 / ``F.normalize`` — the
LayerNorm spreads information across dimensions and discourages dimensional
collapse. The loss always compares predicted latents to *stop-gradient* target
latents.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _norm_target(z: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "layernorm":
        return F.layer_norm(z, (z.shape[-1],))
    if mode == "l2":
        return F.normalize(z, dim=-1)
    if mode == "none":
        return z
    raise ValueError(f"unknown target_norm: {mode}")


class _LatentLoss(nn.Module):
    def __init__(self, target_norm: str = "layernorm") -> None:
        super().__init__()
        self.target_norm = target_norm

    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        target = _norm_target(z_target.detach(), self.target_norm)
        return self._distance(z_pred, target)


class LatentL1Loss(_LatentLoss):
    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class LatentMSELoss(_LatentLoss):
    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class LatentSmoothL1Loss(_LatentLoss):
    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss(pred, target)


class LatentCosineLoss(_LatentLoss):
    def _distance(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()


_LOSSES = {
    "l1": LatentL1Loss,
    "mse": LatentMSELoss,
    "smooth_l1": LatentSmoothL1Loss,
    "cosine": LatentCosineLoss,
}


def build_loss(name: str, target_norm: str = "layernorm") -> _LatentLoss:
    if name not in _LOSSES:
        raise ValueError(f"unknown loss {name!r}; choices: {list(_LOSSES)}")
    return _LOSSES[name](target_norm=target_norm)
