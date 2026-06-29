"""Generate the qualitative / quantitative figures (Figs 5-9, Section 4).

Reads the JSON artefacts produced by the other scripts and renders:

    Fig 5  baseline metric comparison        (from runs/baselines/baseline_table.json)
    Fig 6  ablation deltas vs the full model  (from runs/ablations/ablation_table.json)
    Fig 7  physics-gate weight g over training (from a run's history.json)
    Fig 8  forecast-error-guided resampling distribution evolution
    Fig 9  generated vs real example scenarios

Figures whose inputs are missing are skipped with a notice.  Matplotlib uses a
non-interactive backend so this runs headless.

Usage:
    PYTHONPATH=. python scripts/make_figures.py --smoke
    PYTHONPATH=. python scripts/make_figures.py --baselines runs/baselines/baseline_table.json \
        --ablations runs/ablations/ablation_table.json --history runs/default/history.json
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.config import smoke_config
from src.utils.seed import set_seed
from src.data.datasets import prepare_data
from src.training.trainer import Trainer, AblationFlags

METRIC_COLS = ["ES", "MMD", "Tail_ES", "feasibility_rate",
               "CRPS_reduction_overall", "CRPS_reduction_hard"]


def _load(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def fig_table_bars(table, title, out):
    names = list(table)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), METRIC_COLS):
        vals = [table[n][col] for n in names]
        ax.bar(range(len(names)), vals, color="#4C72B0")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(col)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def fig_gate_curve(history, out):
    if not history or "gate_g" not in history[0]:
        print("  [skip] gate curve: no history")
        return
    g = [h.get("gate_g", np.nan) for h in history]
    ep = [h.get("epoch", i) for i, h in enumerate(history)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep, g, color="#C44E52")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean gate weight g")
    ax.set_title("Fig 7: Physics-gate weight over training")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")


def fig_resampling(trainer, out):
    state = trainer.cond_sampler.state()
    p = np.asarray(state["p"] if isinstance(state, dict) and "p" in state else state)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(p)), p, color="#55A868")
    ax.set_xlabel("meteorological cluster c_ext"); ax.set_ylabel("p(c_ext)")
    ax.set_title("Fig 8: Forecast-error-guided resampling distribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")


def fig_scenarios(trainer, bundle, out):
    c = torch.zeros(4, dtype=torch.long, device=trainer.device)
    gen = trainer.generate(c).cpu().numpy()
    real = bundle.test.power[:4].cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, sharey=True)
    for j in range(4):
        axes[0, j].plot(real[j])
        axes[0, j].set_title(f"real #{j}")
        axes[1, j].plot(gen[j])
        axes[1, j].set_title(f"generated #{j}")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("real"); axes[1, 0].set_ylabel("generated")
    fig.suptitle("Fig 9: Generated vs real multi-site power scenarios")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baselines", type=str, default="runs/baselines/baseline_table.json")
    ap.add_argument("--ablations", type=str, default="runs/ablations/ablation_table.json")
    ap.add_argument("--history", type=str, default="runs/default/history.json")
    ap.add_argument("--out", type=str, default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    bl = _load(args.baselines)
    ab = _load(args.ablations)
    hist = _load(args.history)

    if bl:
        fig_table_bars(bl, "Fig 5: Main comparison (Table 4)",
                       os.path.join(args.out, "fig5_baselines.png"))
    else:
        print("  [skip] Fig 5: no baseline table")
    if ab:
        fig_table_bars(ab, "Fig 6: Ablation study (Table 5)",
                       os.path.join(args.out, "fig6_ablations.png"))
    else:
        print("  [skip] Fig 6: no ablation table")

    # gate curve + qualitative figures need a trained model; in smoke mode we
    # train a tiny one on the fly so the figure pipeline is exercised end-to-end.
    if args.smoke or hist is None:
        cfg = smoke_config()
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name("full"))
        trainer.fit(verbose=False)
        hist = trainer.history
        fig_gate_curve(hist, os.path.join(args.out, "fig7_gate.png"))
        fig_resampling(trainer, os.path.join(args.out, "fig8_resampling.png"))
        fig_scenarios(trainer, bundle, os.path.join(args.out, "fig9_scenarios.png"))
    else:
        fig_gate_curve(hist, os.path.join(args.out, "fig7_gate.png"))
        print("  [note] Figs 8-9 need a live model; run with --smoke or extend "
              "to load a checkpoint for qualitative plots.")


if __name__ == "__main__":
    main()
