"""EMA target encoder utilities.

Only the context encoder and predictor receive gradients. The target encoder is
a full copy of the context encoder (including its patch embedding) updated by an
exponential moving average after each optimizer step.
"""

from __future__ import annotations

import torch
from torch import nn


def init_target_encoder(context_encoder: nn.Module, target_encoder: nn.Module) -> None:
    """Copy context weights into the target encoder and freeze its gradients."""
    target_encoder.load_state_dict(context_encoder.state_dict())
    for p in target_encoder.parameters():
        p.requires_grad_(False)


@torch.no_grad()
def update_ema(context_encoder: nn.Module, target_encoder: nn.Module, momentum: float) -> None:
    """``phi = m * phi + (1 - m) * theta`` for params and buffers."""
    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)
    # Keep buffers (e.g. none here, but future-proof) in sync by copy.
    for b_ctx, b_tgt in zip(context_encoder.buffers(), target_encoder.buffers()):
        b_tgt.data.copy_(b_ctx.data)


class EMAMomentumSchedule:
    """Momentum schedule: ``linear`` ramp start->end, or ``constant``."""

    def __init__(self, start: float, end: float, max_steps: int, schedule: str = "linear") -> None:
        self.start = start
        self.end = end
        self.max_steps = max(max_steps, 1)
        self.schedule = schedule

    def __call__(self, step: int) -> float:
        if self.schedule == "constant":
            return self.start
        frac = min(step / self.max_steps, 1.0)
        return self.start + (self.end - self.start) * frac
