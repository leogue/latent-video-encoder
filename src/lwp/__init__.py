"""Latent world predictor — action-conditioned dynamics (V-JEPA 2-AC style).

Stage 2 of the latent world model. Takes the **frozen** encoder's latents plus
proprioceptive state and actions, and predicts the next-step latent. Trained by
L1 regression in latent space (teacher-forcing + multi-step rollout), with no
collapse machinery — the targets come from a fixed encoder, so this is plain
supervised regression of dynamics.
"""

from __future__ import annotations

from .config import WorldModelConfig, build_world_model, load_config
from .dataset import (
    LeRobotTrajectoryDataset,
    SyntheticTrajectoryDataset,
    build_trajectory_dataset,
)
from .losses import latent_l1
from .model import LatentWorldModel, WorldModelOutput
from .predictor import ActionConditionedPredictor

__all__ = [
    "WorldModelConfig",
    "build_world_model",
    "load_config",
    "LeRobotTrajectoryDataset",
    "SyntheticTrajectoryDataset",
    "build_trajectory_dataset",
    "latent_l1",
    "LatentWorldModel",
    "WorldModelOutput",
    "ActionConditionedPredictor",
]
