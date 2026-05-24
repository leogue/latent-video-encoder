"""LatentWorldModel — frozen encoder + action-conditioned predictor.

Ties Stage 1 (frozen `VideoTransformerEncoder`) to Stage 2 (the predictor).
Produces both the teacher-forcing loss (predict each step from ground-truth
past) and the multi-step rollout loss (autoregressively feed predictions),
combined as in V-JEPA 2-AC.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lve.models import VideoTransformerEncoder

from .losses import latent_l1
from .predictor import ActionConditionedPredictor


@dataclass
class WorldModelOutput:
    loss: torch.Tensor
    loss_tf: torch.Tensor
    loss_rollout: torch.Tensor
    z: torch.Tensor  # [B, T, Hs, D] encoder latents
    pred_tf: torch.Tensor  # [B, T, Hs, D] one-step predictions


class LatentWorldModel(nn.Module):
    def __init__(
        self,
        encoder: VideoTransformerEncoder,
        predictor: ActionConditionedPredictor,
        rollout_steps: int = 4,
        rollout_weight: float = 1.0,
        tf_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.predictor = predictor
        self.rollout_steps = rollout_steps
        self.rollout_weight = rollout_weight
        self.tf_weight = tf_weight

        tg, hg, wg = encoder.patch_embed.grid_shape
        self.num_steps = tg
        self.num_spatial = hg * wg

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """``video`` ``[B, C, T, H, W]`` -> latents ``[B, Tg, Hs, D]`` (t-major)."""
        out = self.encoder.embed(video)
        b, n, d = out.tokens.shape
        return out.tokens.reshape(b, self.num_steps, self.num_spatial, d)

    def trainable_parameters(self):
        return list(self.predictor.parameters())

    def forward(
        self, video: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> WorldModelOutput:
        z = self.encode(video)  # [B, T, Hs, D]

        # Teacher-forcing: predict step t+1 from ground-truth steps <= t.
        pred = self.predictor(z, state, action)  # [B, T, Hs, D]
        loss_tf = latent_l1(pred[:, :-1], z[:, 1:])

        # Multi-step rollout: feed predictions back in autoregressively.
        loss_rollout = self._rollout_loss(z, state, action)

        loss = self.tf_weight * loss_tf + self.rollout_weight * loss_rollout
        return WorldModelOutput(loss, loss_tf, loss_rollout, z, pred)

    def _rollout_loss(
        self, z: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        k = min(self.rollout_steps, z.shape[1] - 1)
        if k <= 0:
            return z.new_zeros(())
        buf = z[:, :1]  # [B, 1, Hs, D] — real first step only
        preds = []
        for t in range(k):
            pred = self.predictor(buf, state[:, : t + 1], action[:, : t + 1])
            next_z = pred[:, t]  # prediction of step t+1  [B, Hs, D]
            preds.append(next_z)
            buf = torch.cat([buf, next_z.unsqueeze(1)], dim=1)
        preds = torch.stack(preds, dim=1)  # [B, k, Hs, D]
        return latent_l1(preds, z[:, 1 : k + 1])

    @torch.no_grad()
    def action_sensitivity(
        self, video: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> float:
        """How much the prediction moves when actions are zeroed (0 = ignored).

        A world model that ignores actions (the "persistence shortcut") scores
        near 0. Watch this during training the way Stage 1 watches collapse —
        a low value means the dynamics aren't action-conditioned.
        """
        z = self.encode(video)
        pred_real = self.predictor(z, state, action)
        pred_zero = self.predictor(z, state, torch.zeros_like(action))
        delta = (pred_real - pred_zero).abs().mean()
        scale = pred_real.abs().mean().clamp(min=1e-6)
        return (delta / scale).item()

    @torch.no_grad()
    def rollout(
        self, z0: torch.Tensor, state: torch.Tensor, action: torch.Tensor, steps: int
    ) -> torch.Tensor:
        """Autoregressively predict ``steps`` future latents from an initial latent.

        Args:
            z0: initial latent ``[B, 1, Hs, D]``.
            state, action: ``[B, steps, *]`` sequences to condition on.
            steps: number of future steps to predict.

        Returns:
            ``[B, steps, Hs, D]`` predicted latents.
        """
        buf = z0
        preds = []
        for t in range(steps):
            pred = self.predictor(buf, state[:, : t + 1], action[:, : t + 1])
            next_z = pred[:, t]
            preds.append(next_z)
            buf = torch.cat([buf, next_z.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)
