"""Config dataclasses, YAML loader, and factory for the world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lve.config import ModelConfig, build_encoder
from lve.probe import load_exported_encoder

from .model import LatentWorldModel
from .predictor import ActionConditionedPredictor


@dataclass
class EncoderConfig:
    # If `path` is set, load a frozen exported encoder; otherwise build a fresh
    # (randomly-initialized, frozen) encoder from these dims — handy for the
    # synthetic backend and tests.
    path: str = ""
    image_size: int = 64
    num_frames: int = 16
    tubelet_size: tuple[int, int, int] = (2, 16, 16)
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    pos_embed_type: str = "rope_3d"


@dataclass
class WMDataConfig:
    backend: str = "synthetic"  # synthetic | lerobot
    num_frames: int = 16
    image_size: int = 64
    state_dim: int = 2
    action_dim: int = 2
    batch_size: int = 8
    samples_per_epoch: int = 5000
    frame_stride: int = 1
    repo_id: str = ""
    root: str = ""
    camera_key: str = ""


@dataclass
class PredictorConfig:
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop_path_rate: float = 0.0


@dataclass
class WorldConfig:
    rollout_steps: int = 4
    rollout_weight: float = 1.0
    tf_weight: float = 1.0


@dataclass
class WMOptimizerConfig:
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 1000
    max_steps: int = 50000
    min_lr: float = 1.0e-6


@dataclass
class WMTrainingConfig:
    max_steps: int = 50000
    log_every: int = 20
    save_every: int = 1000
    grad_clip_norm: float | None = 1.0
    output_dir: str = "./runs/world"


@dataclass
class WorldModelConfig:
    seed: int = 42
    device: str = "cpu"
    precision: str = "fp32"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    data: WMDataConfig = field(default_factory=WMDataConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    optimizer: WMOptimizerConfig = field(default_factory=WMOptimizerConfig)
    training: WMTrainingConfig = field(default_factory=WMTrainingConfig)


def _merge(dc: Any, data: dict[str, Any]) -> Any:
    for k, v in data.items():
        if not hasattr(dc, k):
            raise ValueError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if isinstance(cur, tuple) and isinstance(v, list):
            v = tuple(v)
        setattr(dc, k, v)
    return dc


def load_config(path: str | Path) -> WorldModelConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = WorldModelConfig()
    for key in ("seed", "device", "precision"):
        if key in raw:
            setattr(cfg, key, raw[key])
    for key, dc in (
        ("encoder", cfg.encoder),
        ("data", cfg.data),
        ("predictor", cfg.predictor),
        ("world", cfg.world),
        ("optimizer", cfg.optimizer),
        ("training", cfg.training),
    ):
        if key in raw:
            _merge(dc, raw[key])
    return cfg


def build_frozen_encoder(ec: EncoderConfig):
    """Load an exported encoder if `path` is set, else build a fresh frozen one."""
    if ec.path:
        return load_exported_encoder(ec.path)
    mc = ModelConfig(
        image_size=ec.image_size,
        num_frames=ec.num_frames,
        tubelet_size=tuple(ec.tubelet_size),
        embed_dim=ec.embed_dim,
        depth=ec.depth,
        num_heads=ec.num_heads,
        pos_embed_type=ec.pos_embed_type,
        drop_path_rate=0.0,
    )
    encoder = build_encoder(mc)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def build_world_model(cfg: WorldModelConfig) -> LatentWorldModel:
    encoder = build_frozen_encoder(cfg.encoder)
    tg, hg, wg = encoder.patch_embed.grid_shape
    predictor = ActionConditionedPredictor(
        encoder_dim=encoder.embed_dim,
        state_dim=cfg.data.state_dim,
        action_dim=cfg.data.action_dim,
        num_spatial=hg * wg,
        embed_dim=cfg.predictor.embed_dim,
        depth=cfg.predictor.depth,
        num_heads=cfg.predictor.num_heads,
        mlp_ratio=cfg.predictor.mlp_ratio,
        max_steps=max(tg, 1),
        drop_path_rate=cfg.predictor.drop_path_rate,
    )
    return LatentWorldModel(
        encoder,
        predictor,
        rollout_steps=cfg.world.rollout_steps,
        rollout_weight=cfg.world.rollout_weight,
        tf_weight=cfg.world.tf_weight,
    )
