from __future__ import annotations

import pytest
import torch

from lve.data import SyntheticVideoDataset, clip_delta_timestamps


def test_clip_delta_timestamps_spacing():
    # 16 frames, stride 4, 25 fps -> effective 6.25 fps, step 0.16s.
    dt = clip_delta_timestamps(clip_frames=16, frame_stride=4, fps=25.0)
    assert len(dt) == 16
    assert dt[0] == 0.0
    assert dt[1] == pytest.approx(0.16)
    assert dt[-1] == pytest.approx(0.16 * 15)
    # Monotonic increasing.
    assert all(b > a for a, b in zip(dt, dt[1:]))


def test_clip_delta_timestamps_stride1_is_native():
    dt = clip_delta_timestamps(clip_frames=8, frame_stride=1, fps=10.0)
    assert dt == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def test_synthetic_dataset_shape():
    ds = SyntheticVideoDataset(length=4, clip_frames=8, image_size=64)
    assert ds[0]["video"].shape == (3, 8, 64, 64)


def test_synthetic_dataset_eval_is_deterministic():
    # train=True applies random crop/flip; only the eval transform is deterministic.
    ds = SyntheticVideoDataset(length=4, clip_frames=8, image_size=64, train=False)
    assert torch.equal(ds[0]["video"], ds[0]["video"])
