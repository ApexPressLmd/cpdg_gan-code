"""Hyper-parameter search over the Appendix A grid (Tables A1-A2).

Performs the coordinate-wise sweep around the optimal configuration (or the
full Cartesian product with ``--full``), selecting the configuration with the
best **validation CRPS** of the downstream forecaster trained on real+synthetic
data -- the selection criterion stated in Section 3.6 of the paper.  Each trial
trains the full model, synthesises an augmentation pool, fits the forecaster on
real+synthetic, and scores CRPS on the validation split (lower is better).

Usage:
    PYTHONPATH=. python scripts/hpo.py --smoke
    PYTHONPATH=. python scripts/hpo.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.utils.config import Config, smoke_config
from src.utils.seed import set_seed
from src.data.datasets import prepare_data
from src.training.trainer import Trainer, AblationFlags
from src.models.forecaster import Forecaster, train_forecaster, evaluate_crps
from configs.search_space import iter_grid, apply_override


def _validation_crps(trainer, bundle, cfg) -> float:
    """Validation CRPS of a forecaster trained on real+synthetic (Section 3.6)."""
    device = cfg.device
    real_train = bundle.train.power.to(device)
    n_pool = min(len(bundle.train), 512)
    draw = np.random.choice(trainer.n_clusters, size=n_pool)
    c = torch.as_tensor(draw, dtype=torch.long, device=device)
    with torch.no_grad():
        synth = trainer.generate(c)
    fc = Forecaster(bundle.T, bundle.M).to(device)
    fc = train_forecaster(fc, torch.cat([real_train, synth], dim=0),
                          epochs=cfg.eval.forecaster_epochs, device=device)
    return evaluate_crps(fc, bundle.val.power, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true", help="full Cartesian product")
    ap.add_argument("--max-trials", type=int, default=0, help="0 = no cap")
    ap.add_argument("--out", type=str, default="runs/hpo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    trials = list(iter_grid(full_cartesian=args.full))
    if args.max_trials > 0:
        trials = trials[:args.max_trials]

    log = []
    best = None
    for i, override in enumerate(trials):
        cfg = smoke_config() if args.smoke else Config.load(args.config)
        apply_override(cfg, override)
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name("full"))
        trainer.fit(verbose=False)
        # selection criterion: validation CRPS (lower is better), Section 3.6
        val_crps = _validation_crps(trainer, bundle, cfg)
        log.append({"trial": i, "override": override, "val_CRPS": val_crps})
        print(f"[trial {i:3d}] val_CRPS={val_crps:.4f}  {override}")
        if best is None or val_crps < best["val_CRPS"]:
            best = log[-1]

    with open(os.path.join(args.out, "hpo_log.json"), "w") as fh:
        json.dump({"trials": log, "best": best}, fh, indent=2)
    print(f"\nBEST: val_CRPS={best['val_CRPS']:.4f}  {best['override']}")


if __name__ == "__main__":
    main()
