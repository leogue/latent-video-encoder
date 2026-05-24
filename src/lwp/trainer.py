"""Training loop for the action-conditioned world predictor.

Only the predictor is trained; the encoder stays frozen.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import WorldModelConfig, build_world_model
from .dataset import build_trajectory_dataset

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _infinite(loader: DataLoader):
    while True:
        yield from loader


class WorldModelTrainer:
    def __init__(self, cfg: WorldModelConfig) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.precision = cfg.precision
        self.amp_dtype = _DTYPES[cfg.precision]

        self.model = build_world_model(cfg).to(self.device)
        self.tubelet_t = self.model.encoder.patch_embed.tubelet_size[0]

        dataset = build_trajectory_dataset(cfg.data, self.tubelet_t)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay,
        )
        use_fp16 = cfg.precision == "fp16"
        self.scaler = torch.amp.GradScaler(enabled=use_fp16)
        self.step = 0

    def _lr_at(self, step: int) -> float:
        o = self.cfg.optimizer
        if step < o.warmup_steps:
            return o.lr * step / max(o.warmup_steps, 1)
        frac = (step - o.warmup_steps) / max(o.max_steps - o.warmup_steps, 1)
        frac = min(frac, 1.0)
        return o.min_lr + 0.5 * (o.lr - o.min_lr) * (1 + math.cos(math.pi * frac))

    def _step(self, batch: dict) -> dict[str, float]:
        video = batch["video"].to(self.device)
        state = batch["state"].to(self.device)
        action = batch["action"].to(self.device)

        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.precision != "fp32"
        ):
            out = self.model(video, state, action)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(out.loss).backward()
        grad_norm = torch.tensor(0.0)
        if self.cfg.training.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), self.cfg.training.grad_clip_norm
            )
        lr = self._lr_at(self.step)
        for grp in self.optimizer.param_groups:
            grp["lr"] = lr
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "train/loss": out.loss.item(),
            "train/loss_tf": out.loss_tf.item(),
            "train/loss_rollout": float(out.loss_rollout),
            "train/lr": lr,
            "train/grad_norm": float(grad_norm),
            "train/z_norm": out.z.norm(dim=-1).mean().item(),
        }

    def train(self) -> None:
        self.model.predictor.train()
        out_dir = Path(self.cfg.training.output_dir)
        data_iter = _infinite(self.loader)
        while self.step < self.cfg.training.max_steps:
            batch = next(data_iter)
            metrics = self._step(batch)
            self.step += 1
            if self.step % self.cfg.training.log_every == 0:
                # Diagnostic: are actions actually used? (near 0 = persistence collapse)
                self.model.predictor.eval()
                metrics["train/action_sensitivity"] = self.model.action_sensitivity(
                    batch["video"].to(self.device),
                    batch["state"].to(self.device),
                    batch["action"].to(self.device),
                )
                self.model.predictor.train()
                msg = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items())
                print(f"step {self.step:>7} | {msg}")
            if self.step % self.cfg.training.save_every == 0:
                self.save_checkpoint(out_dir / "checkpoints" / "latest.pt")
        self.save_checkpoint(out_dir / "checkpoints" / "latest.pt")

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": self.step,
                "predictor": self.model.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )
