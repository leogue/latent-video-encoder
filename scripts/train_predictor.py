"""Train the action-conditioned world predictor (Stage 2).

    python scripts/train_predictor.py --config configs/predictor.yaml

The encoder is frozen: set `encoder.path` to an exported encoder.pt (Stage 1),
or leave it empty to use a fresh frozen encoder (synthetic / debugging).
"""

from __future__ import annotations

import argparse

from lwp.config import load_config
from lwp.trainer import WorldModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    WorldModelTrainer(cfg).train()


if __name__ == "__main__":
    main()
