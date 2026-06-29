"""Evaluate a trained CPDG-GAN model on the full metric suite (Section 4.3).

Reports Energy Score, MMD, Tail-ES, feasibility rate, and downstream CRPS
reduction (overall + hard region), plus an efficiency summary (Table 6).

Usage:
    PYTHONPATH=. python scripts/evaluate.py --run runs/default
    PYTHONPATH=. python scripts/evaluate.py --smoke
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
from src.eval.evaluate import evaluate_model, efficiency_summary


def load_trainer(cfg, bundle, ckpt_path, ablation="full"):
    trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name(ablation))
    state = torch.load(ckpt_path, map_location=cfg.device)
    trainer.G.load_state_dict(state["G"])
    trainer.D.load_state_dict(state["D"])
    trainer.Q.load_state_dict(state["Q"])
    trainer.gate.load_state_dict(state["gate"])
    # rebuild a forecaster for the downstream metric
    trainer.outer_resample_step()
    return trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/default")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ablation", type=str, default="full")
    args = ap.parse_args()

    if args.smoke:
        cfg = smoke_config()
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name(args.ablation))
        trainer.fit(verbose=False)
    else:
        cfg = Config.load(os.path.join(args.run, "config.yaml"))
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = load_trainer(cfg, bundle, os.path.join(args.run, "model.pt"),
                               args.ablation)

    metrics = evaluate_model(trainer, bundle, cfg)
    eff = efficiency_summary(trainer, cfg)
    metrics.update(eff)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k != "hard_clusters"}, indent=2))

    if not args.smoke:
        with open(os.path.join(args.run, "metrics.json"), "w") as fh:
            json.dump({k: (v if not isinstance(v, list) else v)
                       for k, v in metrics.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
