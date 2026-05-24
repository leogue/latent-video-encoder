"""VideoJEPAModel — wires context encoder, EMA target encoder and predictor.

Each encoder owns its patch embedding; the target encoder is a full EMA copy
(patch embed included). Do NOT share a single patch embed between them — that
would break the EMA semantics (targets must come from the slow weights).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..ema import init_target_encoder
from .encoder import VideoTransformerEncoder
from .predictor import LatentPredictor


@dataclass
class JEPAOutput:
    loss: torch.Tensor
    z_pred: torch.Tensor  # [B, N_tgt, D]
    z_target: torch.Tensor  # [B, N_tgt, D]


class VideoJEPAModel(nn.Module):
    def __init__(
        self,
        context_encoder: VideoTransformerEncoder,
        target_encoder: VideoTransformerEncoder,
        predictor: LatentPredictor,
        loss_fn: nn.Module,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        init_target_encoder(context_encoder, target_encoder)

    def forward(
        self,
        video: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> JEPAOutput:
        # Context path (gradients flow).
        z_context = self.context_encoder(video, visible_indices=context_indices)

        # Target path (EMA encoder, stop-gradient).
        with torch.no_grad():
            z_target = self.target_encoder(video, visible_indices=target_indices)

        # Predict target latents from context latents + target positions.
        z_pred = self.predictor(
            context_tokens=z_context.tokens,
            context_coords=z_context.coords,
            target_coords=z_target.coords,
        )

        loss = self.loss_fn(z_pred, z_target.tokens)
        return JEPAOutput(loss=loss, z_pred=z_pred, z_target=z_target.tokens)

    def trainable_parameters(self):
        """Parameters that receive gradients: context encoder + predictor."""
        return list(self.context_encoder.parameters()) + list(self.predictor.parameters())
