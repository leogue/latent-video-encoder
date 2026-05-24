"""Train the V-JEPA video encoder.

    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/tiny.yaml --resume runs/tiny/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse

from lve.config import load_config
from lve.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    trainer.train()


if __name__ == "__main__":
    main()
