"""Custom training loop for V-JEPA encoder pretraining."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, build_jepa
from .data import build_dataset
from .ema import EMAMomentumSchedule, update_ema
from .masking import build_collator
from .optim import LRSchedule, WeightDecaySchedule, build_optimizer

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


class Trainer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.precision = cfg.precision
        self.amp_dtype = _DTYPES[cfg.precision]

        self.model = build_jepa(cfg).to(self.device)
        if cfg.training.compile:
            self.model = torch.compile(self.model)

        self.grid_shape = self.model.context_encoder.patch_embed.grid_shape
        self.collator = build_collator(self.grid_shape, cfg.masking)

        dataset = build_dataset(cfg.data, train=True)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            drop_last=True,
        )

        self.optimizer = build_optimizer(self.model, cfg.optimizer)
        self.lr_sched = LRSchedule(cfg.optimizer, cfg.scheduler)
        self.wd_sched = WeightDecaySchedule(cfg.optimizer, cfg.scheduler.max_steps)
        self.ema_sched = EMAMomentumSchedule(
            cfg.ema.momentum_start, cfg.ema.momentum_end, cfg.scheduler.max_steps, cfg.ema.schedule
        )

        use_fp16 = cfg.precision == "fp16"
        self.scaler = torch.amp.GradScaler(enabled=use_fp16)
        self.step = 0

    def _train_step(self, video: torch.Tensor) -> dict[str, float]:
        cfg = self.cfg
        b = video.shape[0]
        context_idx, target_idx = self.collator(b)
        context_idx = context_idx.to(self.device)
        target_idx = target_idx.to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.precision != "fp32",
        ):
            out = self.model(video, context_idx, target_idx)
            loss = out.loss

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        grad_norm = torch.tensor(0.0)
        if cfg.training.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.trainable_parameters(), cfg.training.grad_clip_norm
            )

        lr = self.lr_sched.apply(self.optimizer, self.step)
        wd = self.wd_sched.apply(self.optimizer, self.step)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        momentum = self.ema_sched(self.step)
        update_ema(self.model.context_encoder, self.model.target_encoder, momentum)

        with torch.no_grad():
            cos = F.cosine_similarity(out.z_pred, out.z_target, dim=-1).mean()
            metrics = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/weight_decay": wd,
                "train/ema_momentum": momentum,
                "train/grad_norm": float(grad_norm),
                "train/mask_context_ratio": context_idx.shape[1] / self.collator.n,
                "train/mask_target_ratio": target_idx.shape[1] / self.collator.n,
                "train/z_pred_norm": out.z_pred.norm(dim=-1).mean().item(),
                "train/z_target_norm": out.z_target.norm(dim=-1).mean().item(),
                "train/pred_target_cosine": cos.item(),
            }
        return metrics

    def train(self) -> None:
        self.model.train()
        data_iter = _infinite(self.loader)
        out_dir = Path(self.cfg.training.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        while self.step < self.cfg.training.max_steps:
            video = next(data_iter)["video"].to(self.device)
            metrics = self._train_step(video)
            self.step += 1

            if self.step % self.cfg.training.log_every == 0:
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
                "context_encoder": self.model.context_encoder.state_dict(),
                "target_encoder": self.model.target_encoder.state_dict(),
                "predictor": self.model.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.context_encoder.load_state_dict(ckpt["context_encoder"])
        self.model.target_encoder.load_state_dict(ckpt["target_encoder"])
        self.model.predictor.load_state_dict(ckpt["predictor"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
