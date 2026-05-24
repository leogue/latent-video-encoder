"""Export the frozen EMA (target) encoder — the deliverable of this repo.

    python scripts/export_encoder.py --checkpoint runs/tiny/checkpoints/latest.pt --out encoder.pt

The export is self-contained: a consumer (the world-model stage) can rebuild the
encoder and call ``embed(video)`` without any training code.
"""

from __future__ import annotations

import argparse

import torch

from lve.config import Config, ModelConfig
from lve.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="encoder.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    mc: ModelConfig = cfg_dict["model"]

    export = {
        "encoder": ckpt["target_encoder"],  # EMA weights: stable, higher quality
        "model_config": mc.__dict__,
        "pos_embed_type": mc.pos_embed_type,
        "preprocessing": {
            "image_size": mc.image_size,
            "num_frames": mc.num_frames,
            "tubelet_size": list(mc.tubelet_size),
            "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        },
    }
    torch.save(export, args.out)
    print(f"exported EMA encoder -> {args.out}")


if __name__ == "__main__":
    main()
