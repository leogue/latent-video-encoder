from __future__ import annotations

import torch

from lve.masking import BlockSpec, MultiBlockMaskCollator

GRID = (8, 14, 14)
SPECS = [
    BlockSpec(8, (0.15, 0.15), (1.0, 1.0), (0.75, 1.5)),
    BlockSpec(2, (0.7, 0.7), (1.0, 1.0), (0.75, 1.5)),
]


def _collator():
    return MultiBlockMaskCollator(GRID, SPECS, generator=torch.Generator().manual_seed(0))


def test_shapes_and_batch_sharing():
    col = _collator()
    ctx, tgt = col(batch_size=4)
    assert ctx.shape[0] == 4 and tgt.shape[0] == 4
    # Same mask broadcast across the batch.
    assert torch.equal(ctx[0], ctx[1]) and torch.equal(tgt[0], tgt[3])


def test_indices_in_range():
    col = _collator()
    ctx, tgt = col(batch_size=2)
    n = GRID[0] * GRID[1] * GRID[2]
    assert ctx.min() >= 0 and ctx.max() < n
    assert tgt.min() >= 0 and tgt.max() < n


def test_context_and_target_disjoint():
    col = _collator()
    ctx, tgt = col(batch_size=1)
    ctx_set = set(ctx[0].tolist())
    tgt_set = set(tgt[0].tolist())
    assert ctx_set.isdisjoint(tgt_set)
    assert len(ctx_set) == ctx.shape[1]  # no duplicates


def test_context_is_complement():
    col = _collator()
    ctx, tgt = col(batch_size=1)
    n = GRID[0] * GRID[1] * GRID[2]
    assert ctx.shape[1] + tgt.shape[1] == n


def test_temporal_full_extent_by_default():
    # temporal_scale 1.0 -> every block spans all frames, so if a (y,x) cell is
    # masked at one t it is masked at all t. Check the target set is consistent.
    col = _collator()
    _, tgt = col(batch_size=1)
    tg, hg, wg = GRID
    spatial = {(idx % (hg * wg)) for idx in tgt[0].tolist()}
    # Reconstruct: each masked spatial cell should appear at all tg time steps.
    counts: dict[int, int] = {}
    for idx in tgt[0].tolist():
        counts[idx % (hg * wg)] = counts.get(idx % (hg * wg), 0) + 1
    assert all(c == tg for c in counts.values())
    assert len(spatial) > 0


def test_max_keep_caps_context():
    col = MultiBlockMaskCollator(
        GRID, SPECS, max_keep=50, generator=torch.Generator().manual_seed(1)
    )
    ctx, _ = col(batch_size=1)
    assert ctx.shape[1] <= 50
