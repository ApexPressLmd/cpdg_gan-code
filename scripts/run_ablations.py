"""Ablation study reproducing Table 5 (Section 4.4).

Trains and evaluates the full model and each ablated variant on the same data
split and reports the metric suite for every row.  The "fusion-without-Delta"
comparison row is the ``-delta1delta2`` variant.

Variants:
    full           the complete CPDG-GAN
    -delta1        remove the diagnostic-driven physics gate (fixed-weight reg)
    -delta2        remove forecast-error-guided condition resampling (uniform)
    -delta1delta2  fusion architecture without either innovation
    -a6            remove controllable internal axes + MI
    -a9            remove meteorological-causal conditioning (single cluster)

Usage:
    PYTHONPATH=. python scripts/run_ablations.py --config configs/default.yaml
    PYTHONPATH=. python scripts/run_ablations.py --smoke
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

VARIANTS = ["full", "-delta1", "-delta2", "-delta1delta2", "-a6", "-a9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--out", type=str, default="runs/ablations")
    args = ap.parse_args()

    base_cfg = smoke_config() if args.smoke else Config.load(args.config)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for name in args.variants:
        print(f"\n========== ablation: {name} ==========")
        cfg = smoke_config() if args.smoke else Config.load(args.config)
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name(name))
        trainer.fit(verbose=False)
        m = evaluate_model(trainer, bundle, cfg)
        m.pop("hard_clusters", None)
        results[name] = m

    with open(os.path.join(args.out, "ablation_table.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("\n=== Table 5 (ablations) ===")
    _print_table(results)


def _print_table(results):
    cols = ["ES", "MMD", "Tail_ES", "feasibility_rate",
            "CRPS_reduction_overall", "CRPS_reduction_hard"]
    header = "variant".ljust(16) + "".join(c[:10].rjust(12) for c in cols)
    print(header)
    for name, m in results.items():
        row = name.ljust(16) + "".join(f"{m[c]:12.4f}" for c in cols)
        print(row)


if __name__ == "__main__":
    main()
