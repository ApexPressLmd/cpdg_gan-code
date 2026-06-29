"""Baseline comparison reproducing Tables 3-4 (Sections 4.2-4.3).

Trains and evaluates every baseline in :data:`baselines.REGISTRY` plus the full
CPDG-GAN ("Ours") and the fusion-without-Delta variant, on a common data split,
and reports the metric suite for each.

Usage:
    PYTHONPATH=. python scripts/run_baselines.py --config configs/default.yaml
    PYTHONPATH=. python scripts/run_baselines.py --smoke
    PYTHONPATH=. python scripts/run_baselines.py --smoke --only wgan_gp timegan
"""
from __future__ import annotations

import argparse
import json
import os

from src.utils.config import Config, smoke_config
from src.utils.seed import set_seed
from src.data.datasets import prepare_data
from src.training.trainer import Trainer, AblationFlags
from src.eval.evaluate import evaluate_model
from baselines import REGISTRY, build_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of baseline keys to run")
    ap.add_argument("--out", type=str, default="runs/baselines")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    keys = args.only if args.only else list(REGISTRY)
    results = {}

    # ---- baselines --------------------------------------------------------
    for key in keys:
        print(f"\n========== baseline: {key} ==========")
        cfg = smoke_config() if args.smoke else Config.load(args.config)
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        model = build_baseline(key, cfg, bundle, device=cfg.device)
        model.fit(verbose=False)
        m = evaluate_model(model, bundle, cfg)
        m.pop("hard_clusters", None)
        results[getattr(model, "name", key)] = m

    # ---- fusion-without-Delta + full model (for the same table) -----------
    for tag, ablation in [("Fusion (no Delta)", "-delta1delta2"), ("CPDG-GAN (Ours)", "full")]:
        print(f"\n========== {tag} ==========")
        cfg = smoke_config() if args.smoke else Config.load(args.config)
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name(ablation))
        trainer.fit(verbose=False)
        m = evaluate_model(trainer, bundle, cfg)
        m.pop("hard_clusters", None)
        results[tag] = m

    with open(os.path.join(args.out, "baseline_table.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n=== Tables 3-4 (baseline comparison) ===")
    cols = ["ES", "MMD", "Tail_ES", "feasibility_rate",
            "CRPS_reduction_overall", "CRPS_reduction_hard"]
    print("model".ljust(20) + "".join(c[:10].rjust(12) for c in cols))
    for name, m in results.items():
        print(name.ljust(20) + "".join(f"{m[c]:12.4f}" for c in cols))


if __name__ == "__main__":
    main()
