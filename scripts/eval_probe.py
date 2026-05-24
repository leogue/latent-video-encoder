"""Evaluate a frozen encoder with an attentive probe.

    python scripts/eval_probe.py --encoder encoder.pt --epochs 10

Expects a labeled clip dataset yielding ``{"video": [C,T,H,W], "label": int}``.
Plug your dataset into ``build_labeled_dataset`` below; only the probe head is
trained, the encoder stays frozen.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from lve.probe import AttentiveProbe, load_exported_encoder


def build_labeled_dataset(train: bool):
    raise NotImplementedError(
        "Provide a labeled clip dataset (e.g. Something-Something v2, Kinetics-400, "
        "Diving-48) returning {'video': [C,T,H,W], 'label': int}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = load_exported_encoder(args.encoder).to(device)
    probe = AttentiveProbe(encoder.embed_dim, args.num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    ce = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(build_labeled_dataset(train=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(build_labeled_dataset(train=False), batch_size=args.batch_size)

    for epoch in range(args.epochs):
        probe.train()
        for batch in train_loader:
            video = batch["video"].to(device)
            label = batch["label"].to(device)
            with torch.no_grad():
                tokens = encoder.embed(video).tokens
            logits = probe(tokens)
            loss = ce(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device)
                label = batch["label"].to(device)
                logits = probe(encoder.embed(video).tokens)
                correct += (logits.argmax(-1) == label).sum().item()
                total += label.numel()
        print(f"epoch {epoch}: val top-1 = {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
