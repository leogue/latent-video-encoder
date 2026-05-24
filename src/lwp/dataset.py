"""Trajectory datasets for the action-conditioned world model.

Each sample is ``{"video": [C, T, H, W], "state": [Tg, state_dim],
"action": [Tg, action_dim]}`` where ``Tg = T // tubelet_t`` is the number of
encoder timesteps. State/action are sampled at the *step* rate (one per tubelet),
the video at the *frame* rate.

Geometry-changing augmentations are disabled: random crops/flips would break the
correspondence between actions and the observed motion.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from lve.data.transforms import VideoTransform


class SyntheticTrajectoryDataset(Dataset):
    """A square whose motion is *controlled by the action* — a learnable,
    action-conditioned toy world.

    Dynamics: ``pos_{t+1} = clamp(pos_t + action_t)``. The action is the
    per-step velocity, the state is the position. A model that ignores the
    action cannot predict the next latent — making this a direct test that the
    predictor is action-conditioned.
    """

    def __init__(
        self,
        length: int = 512,
        num_frames: int = 16,
        tubelet_t: int = 2,
        image_size: int = 64,
        max_speed: float = 0.12,
        seed: int = 0,
    ) -> None:
        assert num_frames % tubelet_t == 0
        self.length = length
        self.num_frames = num_frames
        self.tubelet_t = tubelet_t
        self.num_steps = num_frames // tubelet_t
        self.image_size = image_size
        self.max_speed = max_speed
        self.seed = seed
        self.transform = VideoTransform(image_size, train=False)  # geometry-preserving

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed + idx)
        size, sq = self.image_size, self.image_size // 4
        span = 1.0  # positions live in [0, 1]
        color = torch.rand(3, generator=g)

        pos = torch.rand(2, generator=g)  # (x, y) in [0,1]
        positions, actions = [], []
        for _ in range(self.num_steps):
            a = (torch.rand(2, generator=g) * 2 - 1) * self.max_speed
            positions.append(pos.clone())
            actions.append(a.clone())
            pos = (pos + a).clamp(0.0, span)
        state = torch.stack(positions)  # [Tg, 2]
        action = torch.stack(actions)  # [Tg, 2]

        clip = torch.zeros(3, self.num_frames, size, size)
        for f in range(self.num_frames):
            p = positions[f // self.tubelet_t]
            x = int(p[0].item() * (size - sq))
            y = int(p[1].item() * (size - sq))
            for c in range(3):
                clip[c, f, y : y + sq, x : x + sq] = color[c]
        return {"video": self.transform(clip), "state": state, "action": action}


class LeRobotTrajectoryDataset(Dataset):
    """Trajectory windows from a real LeRobotDataset (camera + state + action).

    Requires the ``lerobot`` package. Samples ``num_frames`` camera frames plus
    ``num_steps`` proprioceptive states and actions, aligned to the tubelet step
    rate. Loose video files are NOT supported here — use a real LeRobotDataset.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        camera_key: str | None = None,
        state_key: str = "observation.state",
        action_key: str = "action",
        num_frames: int = 16,
        tubelet_t: int = 2,
        frame_stride: int = 1,
        image_size: int = 224,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import (  # lazy, heavy dependency
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'lerobot' world-model backend requires lerobot: `uv add lerobot`"
            ) from e

        assert num_frames % tubelet_t == 0
        self.num_steps = num_frames // tubelet_t
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        cameras = list(meta.camera_keys)
        if not cameras:
            raise RuntimeError(f"no camera keys in {repo_id!r}")
        self.camera_key = camera_key or cameras[0]
        self.state_key, self.action_key = state_key, action_key

        fps = meta.fps
        frame_dt = [round(k * frame_stride / fps, 6) for k in range(num_frames)]
        step_dt = [round(k * tubelet_t * frame_stride / fps, 6) for k in range(self.num_steps)]
        delta = {self.camera_key: frame_dt, state_key: step_dt, action_key: step_dt}
        self.dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
        self.transform = VideoTransform(image_size, train=False)

    def __len__(self) -> int:
        return self.dataset.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        clip = sample[self.camera_key].permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        if clip.shape[0] == 1:
            clip = clip.repeat(3, 1, 1, 1)
        return {
            "video": self.transform(clip),
            "state": sample[self.state_key].float(),  # [Tg, state_dim]
            "action": sample[self.action_key].float(),  # [Tg, action_dim]
        }


def build_trajectory_dataset(cfg, tubelet_t: int):
    """Build a trajectory dataset from a data config object."""
    if cfg.backend == "synthetic":
        return SyntheticTrajectoryDataset(
            length=cfg.samples_per_epoch,
            num_frames=cfg.num_frames,
            tubelet_t=tubelet_t,
            image_size=cfg.image_size,
        )
    if cfg.backend == "lerobot":
        return LeRobotTrajectoryDataset(
            repo_id=cfg.repo_id,
            root=cfg.root or None,
            camera_key=cfg.camera_key or None,
            num_frames=cfg.num_frames,
            tubelet_t=tubelet_t,
            frame_stride=cfg.frame_stride,
            image_size=cfg.image_size,
        )
    raise ValueError(f"unknown world-model data backend {cfg.backend!r}")
