"""Optimizer parameter groups and LR / weight-decay schedules."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import OptimizerConfig, SchedulerConfig


def _no_decay(name: str, param: torch.Tensor) -> bool:
    """No weight decay on biases, norms, position embeds, and mask tokens."""
    if param.ndim <= 1:  # biases, LayerNorm weights
        return True
    keywords = ("pos_embed", "mask_token")
    return any(k in name for k in keywords)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if _no_decay(name, param) else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    assert cfg.name == "adamw", f"only adamw supported, got {cfg.name}"
    groups = build_param_groups(model, cfg.weight_decay)
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas))


class LRSchedule:
    """LR multiplier schedule: warmup then cosine, or warmup-constant-cooldown."""

    def __init__(self, opt_cfg: OptimizerConfig, sched_cfg: SchedulerConfig) -> None:
        self.peak = opt_cfg.lr
        self.start = opt_cfg.start_lr
        self.final = opt_cfg.final_lr
        self.min_lr = sched_cfg.min_lr
        self.warmup = sched_cfg.warmup_steps
        self.cooldown = sched_cfg.cooldown_steps
        self.max_steps = sched_cfg.max_steps
        self.name = sched_cfg.name

    def lr_at(self, step: int) -> float:
        if step < self.warmup:
            frac = step / max(self.warmup, 1)
            return self.start + (self.peak - self.start) * frac
        if self.name == "cosine":
            frac = (step - self.warmup) / max(self.max_steps - self.warmup, 1)
            frac = min(frac, 1.0)
            return self.final + 0.5 * (self.peak - self.final) * (1 + math.cos(math.pi * frac))
        # warmup_constant_decay (trapezoidal)
        cooldown_start = self.max_steps - self.cooldown
        if step < cooldown_start:
            return self.peak
        frac = (step - cooldown_start) / max(self.cooldown, 1)
        frac = min(frac, 1.0)
        return self.peak + (self.final - self.peak) * frac

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = max(self.lr_at(step), self.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr


class WeightDecaySchedule:
    """Optionally ramp weight decay (cosine) from start -> final over training."""

    def __init__(self, opt_cfg: OptimizerConfig, max_steps: int) -> None:
        self.start = opt_cfg.weight_decay
        self.final = opt_cfg.final_weight_decay
        self.max_steps = max(max_steps, 1)

    def apply(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        if self.final is None:
            return self.start
        frac = min(step / self.max_steps, 1.0)
        wd = self.start + (self.final - self.start) * 0.5 * (1 - math.cos(math.pi * frac))
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:  # leave the no-decay group at 0
                group["weight_decay"] = wd
        return wd
