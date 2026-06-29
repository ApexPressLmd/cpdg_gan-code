"""Train the full CPDG-GAN fusion model (Algorithm 1 + Algorithm 2).

Usage:
    PYTHONPATH=. python scripts/train.py --config configs/default.yaml
    PYTHONPATH=. python scripts/train.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from src.utils.config import Config, smoke_config
from src.utils.seed import set_seed
from src.data.datasets import prepare_data
from src.training.trainer import Trainer, AblationFlags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true", help="use the tiny CI config")
    ap.add_argument("--ablation", type=str, default="full",
                    help="full|-delta1|-delta2|-delta1delta2|-a6|-a9")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else Config.load(args.config)
    out_dir = args.out or cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)

    bundle = prepare_data(cfg)
    flags = AblationFlags.from_name(args.ablation)
    trainer = Trainer(cfg, bundle, flags=flags)
    trainer.fit(verbose=not args.quiet)

    ckpt = os.path.join(out_dir, "model.pt")
    trainer.save(ckpt)
    cfg.save(os.path.join(out_dir, "config.yaml"))
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(trainer.history, fh, indent=2)
    print(f"saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
