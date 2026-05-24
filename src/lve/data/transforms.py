"""Clip-consistent video transforms.

The same crop and flip is applied to *every* frame of a clip — never an
independent random transform per frame.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoTransform:
    def __init__(
        self,
        image_size: int,
        train: bool = True,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.image_size = image_size
        self.train = train
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Args: ``clip`` ``[C, T, H, W]`` float in [0, 1]. Returns normalized ``[C, T, H, W]``."""
        c, t, h, w = clip.shape
        size = self.image_size

        if self.train:
            # One random resized crop shared across all frames.
            scale = 0.5 + 0.5 * torch.rand(1).item()
            ch = max(1, int(h * scale))
            cw = max(1, int(w * scale))
            top = torch.randint(0, h - ch + 1, (1,)).item()
            left = torch.randint(0, w - cw + 1, (1,)).item()
            clip = clip[:, :, top : top + ch, left : left + cw]
            if torch.rand(1).item() < 0.5:
                clip = torch.flip(clip, dims=[-1])  # horizontal flip, all frames

        # Resize every frame to the target square via a single interpolate call.
        clip = F.interpolate(clip, size=(size, size), mode="bilinear", align_corners=False)
        return (clip - self.mean) / self.std
