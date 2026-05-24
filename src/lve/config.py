"""YAML config loading, dataclasses, and model/optim factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .losses import build_loss
from .models import LatentPredictor, VideoJEPAModel, VideoTransformerEncoder


@dataclass
class DataConfig:
    root: str = "./data/videos"
    backend: str = "synthetic"  # synthetic | frames | torchvision
    clip_frames: int = 16
    frame_stride: int = 2
    image_size: int = 128
    batch_size: int = 2
    num_workers: int = 0
    samples_per_epoch: int = 1000  # used by the "frames" backend


@dataclass
class ModelConfig:
    in_channels: int = 3
    image_size: int = 128
    num_frames: int = 16
    tubelet_size: tuple[int, int, int] = (2, 16, 16)
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    pos_embed_type: str = "rope_3d"
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    use_cls_token: bool = False


@dataclass
class PredictorConfig:
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    pos_embed_type: str = "rope_3d"


@dataclass
class LossConfig:
    name: str = "l1"
    target_norm: str = "layernorm"
    reg_coeff: float = 0.0


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6.25e-4
    start_lr: float = 2.0e-4
    final_lr: float = 1.0e-6
    weight_decay: float = 0.04
    final_weight_decay: float | None = 0.4
    betas: tuple[float, float] = (0.9, 0.95)


@dataclass
class SchedulerConfig:
    name: str = "warmup_constant_decay"  # or "cosine"
    warmup_steps: int = 1000
    cooldown_steps: int = 1000
    max_steps: int = 20000
    min_lr: float = 1.0e-6


@dataclass
class EMAConfig:
    schedule: str = "linear"
    momentum_start: float = 0.998
    momentum_end: float = 1.0


@dataclass
class TrainingConfig:
    max_steps: int = 20000
    log_every: int = 20
    save_every: int = 1000
    grad_clip_norm: float | None = 1.0
    output_dir: str = "./runs/tiny"
    compile: bool = False


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"  # fp32 | fp16 | bf16
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    masking: dict[str, Any] = field(default_factory=dict)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    """Override dataclass fields from a dict; tuples are coerced from lists."""
    for k, v in data.items():
        if not hasattr(dc, k):
            raise ValueError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(dc, k, v)
    return dc


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = Config()
    cfg.seed = raw.get("seed", cfg.seed)
    cfg.device = raw.get("device", cfg.device)
    cfg.precision = raw.get("precision", cfg.precision)
    if "data" in raw:
        _merge(cfg.data, raw["data"])
    if "model" in raw:
        _merge(cfg.model, raw["model"])
    if "predictor" in raw:
        _merge(cfg.predictor, raw["predictor"])
    if "masking" in raw:
        cfg.masking = raw["masking"]
    if "loss" in raw:
        _merge(cfg.loss, raw["loss"])
    if "optimizer" in raw:
        _merge(cfg.optimizer, raw["optimizer"])
    if "scheduler" in raw:
        _merge(cfg.scheduler, raw["scheduler"])
    if "ema" in raw:
        _merge(cfg.ema, raw["ema"])
    if "training" in raw:
        _merge(cfg.training, raw["training"])

    # Sanity: dataloader must emit exactly what the encoder expects.
    assert cfg.data.clip_frames == cfg.model.num_frames, (
        f"data.clip_frames ({cfg.data.clip_frames}) must equal "
        f"model.num_frames ({cfg.model.num_frames})"
    )
    return cfg


def build_encoder(mc: ModelConfig) -> VideoTransformerEncoder:
    return VideoTransformerEncoder(
        image_size=mc.image_size,
        num_frames=mc.num_frames,
        tubelet_size=tuple(mc.tubelet_size),
        in_channels=mc.in_channels,
        embed_dim=mc.embed_dim,
        depth=mc.depth,
        num_heads=mc.num_heads,
        mlp_ratio=mc.mlp_ratio,
        pos_embed_type=mc.pos_embed_type,
        drop_rate=mc.drop_rate,
        attn_drop_rate=mc.attn_drop_rate,
        drop_path_rate=mc.drop_path_rate,
        use_cls_token=mc.use_cls_token,
    )


def build_jepa(cfg: Config) -> VideoJEPAModel:
    context_encoder = build_encoder(cfg.model)
    # Target encoder uses no stochastic depth (it never backprops).
    target_mc = ModelConfig(**{**cfg.model.__dict__, "drop_path_rate": 0.0})
    target_encoder = build_encoder(target_mc)
    predictor = LatentPredictor(
        encoder_embed_dim=cfg.model.embed_dim,
        embed_dim=cfg.predictor.embed_dim,
        depth=cfg.predictor.depth,
        num_heads=cfg.predictor.num_heads,
        mlp_ratio=cfg.predictor.mlp_ratio,
        pos_embed_type=cfg.predictor.pos_embed_type,
    )
    loss_fn = build_loss(cfg.loss.name, cfg.loss.target_norm)
    return VideoJEPAModel(context_encoder, target_encoder, predictor, loss_fn)
