"""Multi-block spatio-temporal masking (V-JEPA style).

The *same* mask is sampled once and applied to every clip in the batch, so
``N_context`` and ``N_target`` are identical across the batch and can be stacked
into ``[B, N]`` tensors. Target blocks are sampled first; the context is their
complement (the two never overlap).

Block families default to full temporal extent (``temporal_scale = 1.0``): each
block is a spatial region repeated over all frames, which forces the encoder to
reason across time. Causal / future-frame masking is intentionally not provided
here (that belongs to the later action-conditioned stage).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class BlockSpec:
    num_blocks: int
    spatial_scale: tuple[float, float]  # fraction of (Hg*Wg) per block
    temporal_scale: tuple[float, float]  # 1.0 -> spans all frames
    aspect_ratio: tuple[float, float]  # block height:width range


class MultiBlockMaskCollator:
    def __init__(
        self,
        grid_shape: tuple[int, int, int],
        block_specs: list[BlockSpec],
        max_keep: int | None = None,
        min_context: int = 1,
        generator: torch.Generator | None = None,
    ) -> None:
        self.tg, self.hg, self.wg = grid_shape
        self.n = self.tg * self.hg * self.wg
        self.block_specs = block_specs
        self.max_keep = max_keep
        self.min_context = min_context
        self.generator = generator

    def _rand(self, lo: float, hi: float) -> float:
        u = torch.rand(1, generator=self.generator).item()
        return lo + (hi - lo) * u

    def _sample_block(self, spec: BlockSpec) -> torch.Tensor:
        """Boolean ``[Tg, Hg, Wg]`` mask for one block (True = masked/target)."""
        scale = self._rand(*spec.spatial_scale)
        ar = self._rand(*spec.aspect_ratio)
        area = scale * self.hg * self.wg
        bh = int(round(math.sqrt(area * ar)))
        bw = int(round(math.sqrt(area / ar)))
        bh = max(1, min(self.hg, bh))
        bw = max(1, min(self.wg, bw))

        tscale = self._rand(*spec.temporal_scale)
        bt = max(1, min(self.tg, int(round(tscale * self.tg))))

        top = int(self._rand(0, self.hg - bh + 1))
        left = int(self._rand(0, self.wg - bw + 1))
        front = int(self._rand(0, self.tg - bt + 1))

        mask = torch.zeros(self.tg, self.hg, self.wg, dtype=torch.bool)
        mask[front : front + bt, top : top + bh, left : left + bw] = True
        return mask

    def _sample_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one (context_indices, target_indices) pair for the whole batch."""
        for _ in range(100):  # resample until both sides are non-empty
            masked = torch.zeros(self.tg, self.hg, self.wg, dtype=torch.bool)
            for spec in self.block_specs:
                for _ in range(spec.num_blocks):
                    masked |= self._sample_block(spec)
            masked = masked.reshape(-1)
            target_idx = masked.nonzero(as_tuple=False).squeeze(-1)
            context_idx = (~masked).nonzero(as_tuple=False).squeeze(-1)
            if target_idx.numel() > 0 and context_idx.numel() >= self.min_context:
                if self.max_keep is not None and context_idx.numel() > self.max_keep:
                    perm = torch.randperm(context_idx.numel(), generator=self.generator)
                    context_idx = context_idx[perm[: self.max_keep]].sort().values
                return context_idx, target_idx
        raise RuntimeError("failed to sample a valid mask; check block_specs vs grid_shape")

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context_indices, target_indices)``, each ``[B, N]`` with the
        same mask broadcast across the batch."""
        context_idx, target_idx = self._sample_mask()
        context = context_idx.unsqueeze(0).expand(batch_size, -1).contiguous()
        target = target_idx.unsqueeze(0).expand(batch_size, -1).contiguous()
        return context, target


def build_collator(grid_shape: tuple[int, int, int], cfg: dict) -> MultiBlockMaskCollator:
    """Build a collator from a parsed masking config dict."""
    specs = [
        BlockSpec(
            num_blocks=b["num_blocks"],
            spatial_scale=tuple(b["spatial_scale"]),
            temporal_scale=tuple(b["temporal_scale"]),
            aspect_ratio=tuple(b["aspect_ratio"]),
        )
        for b in cfg["blocks"]
    ]
    return MultiBlockMaskCollator(grid_shape, specs, max_keep=cfg.get("max_keep"))
