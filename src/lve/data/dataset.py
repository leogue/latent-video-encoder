"""Video datasets: a synthetic fallback (for tests/dev) and a folder dataset."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .transforms import VideoTransform


class SyntheticVideoDataset(Dataset):
    """Moving colored squares — lets the training loop run with no real videos.

    Returns ``{"video": [C, T, H, W]}`` already normalized.
    """

    def __init__(
        self,
        length: int = 256,
        clip_frames: int = 16,
        image_size: int = 128,
        in_channels: int = 3,
        train: bool = True,
    ) -> None:
        self.length = length
        self.clip_frames = clip_frames
        self.image_size = image_size
        self.in_channels = in_channels
        self.transform = VideoTransform(image_size, train=train)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx)
        size = self.image_size
        t = self.clip_frames
        clip = torch.zeros(self.in_channels, t, size, size)
        # A square of random color moving linearly across the frame.
        sq = size // 4
        color = torch.rand(self.in_channels, generator=g)
        x0 = torch.randint(0, size - sq, (1,), generator=g).item()
        y0 = torch.randint(0, size - sq, (1,), generator=g).item()
        vx = torch.randint(-3, 4, (1,), generator=g).item()
        vy = torch.randint(-3, 4, (1,), generator=g).item()
        for f in range(t):
            x = min(max(x0 + vx * f, 0), size - sq)
            y = min(max(y0 + vy * f, 0), size - sq)
            for c in range(self.in_channels):
                clip[c, f, y : y + sq, x : x + sq] = color[c]
        return {"video": self.transform(clip)}


class VideoFolderDataset(Dataset):
    """Decode random clips from a folder of video files (torchvision backend)."""

    def __init__(
        self,
        root: str,
        clip_frames: int = 16,
        frame_stride: int = 2,
        image_size: int = 224,
        train: bool = True,
        extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm"),
    ) -> None:
        self.root = Path(root)
        self.files = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in extensions)
        if not self.files:
            raise FileNotFoundError(f"no video files under {root}")
        self.clip_frames = clip_frames
        self.frame_stride = frame_stride
        self.transform = VideoTransform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from torchvision.io import read_video  # imported lazily

        path = self.files[idx]
        frames, _, _ = read_video(str(path), output_format="TCHW", pts_unit="sec")
        frames = frames.float() / 255.0  # [T, C, H, W]
        span = self.clip_frames * self.frame_stride
        if frames.shape[0] < span:
            reps = span // max(frames.shape[0], 1) + 1
            frames = frames.repeat(reps, 1, 1, 1)
        start = torch.randint(0, frames.shape[0] - span + 1, (1,)).item()
        clip = frames[start : start + span : self.frame_stride]  # [t, C, H, W]
        clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
        return {"video": self.transform(clip)}


class FrameClipDataset(Dataset):
    """Sample temporal clips from pre-extracted JPG frames (see preprocess_videos.py).

    ``root`` contains one sub-directory per video, each holding ``frame_*.jpg``.
    Because each source video is long, we sample ``samples_per_epoch`` random
    temporal windows (across all videos) rather than one clip per file. Frames
    are read one at a time with torchvision's JPEG decoder — no video decoder
    dependency, memory-safe.
    """

    def __init__(
        self,
        root: str,
        clip_frames: int = 16,
        frame_stride: int = 4,
        image_size: int = 224,
        samples_per_epoch: int = 1000,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.clips = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not self.clips:
            raise FileNotFoundError(f"no frame sub-directories under {root}; run preprocess_videos.py")
        self.frames: list[list[Path]] = []
        span = clip_frames * frame_stride
        for d in self.clips:
            fr = sorted(d.glob("frame_*.jpg"))
            if len(fr) >= span:
                self.frames.append(fr)
        if not self.frames:
            raise RuntimeError(f"no video has >= {span} frames; lower clip_frames/frame_stride or fps")
        self.clip_frames = clip_frames
        self.frame_stride = frame_stride
        self.samples_per_epoch = samples_per_epoch
        self.transform = VideoTransform(image_size, train=train)
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from torchvision.io import read_image  # JPEG decoder, still in torchvision 0.27

        g = torch.Generator().manual_seed(self.seed + idx)
        vid = self.frames[torch.randint(len(self.frames), (1,), generator=g).item()]
        span = self.clip_frames * self.frame_stride
        start = torch.randint(0, len(vid) - span + 1, (1,), generator=g).item()
        sel = vid[start : start + span : self.frame_stride]
        frames = [read_image(str(p)).float() / 255.0 for p in sel]  # each [C, H, W]
        clip = torch.stack(frames, dim=1)  # [C, T, H, W]
        if clip.shape[0] == 1:  # grayscale -> RGB
            clip = clip.repeat(3, 1, 1, 1)
        return {"video": self.transform(clip)}


def clip_delta_timestamps(clip_frames: int, frame_stride: int, fps: float) -> list[float]:
    """Time offsets (seconds, relative to the current frame) for a strided clip.

    ``clip_frames`` offsets spaced ``frame_stride / fps`` apart, so the effective
    sampling rate is ``fps / frame_stride`` (same convention as the frame backend).
    """
    step = frame_stride / fps
    return [round(k * step, 6) for k in range(clip_frames)]


class LeRobotClipDataset(Dataset):
    """Sample temporal video clips from a LeRobotDataset (v2.1 / v3.0).

    Loads a real LeRobot dataset (``meta/`` + Parquet + MP4 shards) from the Hub
    or a local ``root``, picks one camera stream (``observation.images.<cam>``)
    and returns ``{"video": [C, T, H, W]}`` clips. Actions and low-dim state are
    ignored — the encoder is trained on pixels only; actions belong to the later
    action-conditioned world-model stage.

    Requires the ``lerobot`` package (``uv add lerobot``). NOTE: this expects a
    LeRobotDataset, not a folder of loose video files — use the ``frames``
    backend for those.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | None = None,
        camera_key: str | None = None,
        clip_frames: int = 16,
        frame_stride: int = 4,
        image_size: int = 224,
        train: bool = True,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import (  # lazy, heavy dependency
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        except ImportError as e:  # pragma: no cover - depends on optional package
            raise ImportError(
                "the 'lerobot' backend requires the lerobot package: `uv add lerobot`"
            ) from e

        meta = LeRobotDatasetMetadata(repo_id, root=root)
        cameras = list(meta.camera_keys)
        if not cameras:
            raise RuntimeError(f"no camera keys in dataset {repo_id!r}")
        if camera_key in (None, ""):
            camera_key = cameras[0]
        elif camera_key not in cameras:
            raise ValueError(f"camera_key {camera_key!r} not in {cameras}")
        self.camera_key = camera_key

        delta = {camera_key: clip_delta_timestamps(clip_frames, frame_stride, meta.fps)}
        self.dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
        self.transform = VideoTransform(image_size, train=train)

    def __len__(self) -> int:
        return self.dataset.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        clip = self.dataset[idx][self.camera_key]  # [T, C, H, W], float in [0, 1]
        clip = clip.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        if clip.shape[0] == 1:  # grayscale -> RGB
            clip = clip.repeat(3, 1, 1, 1)
        return {"video": self.transform(clip)}


def build_dataset(cfg, train: bool = True) -> Dataset:
    """Build a dataset from a DataConfig-like object."""
    if cfg.backend == "lerobot":
        return LeRobotClipDataset(
            repo_id=cfg.repo_id,
            root=cfg.root or None,
            camera_key=getattr(cfg, "camera_key", "") or None,
            clip_frames=cfg.clip_frames,
            frame_stride=cfg.frame_stride,
            image_size=cfg.image_size,
            train=train,
        )
    if cfg.backend == "synthetic":
        return SyntheticVideoDataset(
            clip_frames=cfg.clip_frames, image_size=cfg.image_size, train=train
        )
    if cfg.backend == "frames":
        return FrameClipDataset(
            root=cfg.root,
            clip_frames=cfg.clip_frames,
            frame_stride=cfg.frame_stride,
            image_size=cfg.image_size,
            samples_per_epoch=getattr(cfg, "samples_per_epoch", 1000),
            train=train,
        )
    if cfg.backend == "torchvision":
        return VideoFolderDataset(
            root=cfg.root,
            clip_frames=cfg.clip_frames,
            frame_stride=cfg.frame_stride,
            image_size=cfg.image_size,
            train=train,
        )
    raise ValueError(f"unknown data backend {cfg.backend!r}")
