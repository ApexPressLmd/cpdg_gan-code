"""End-to-end evaluation of a trained model (Tables 4-6).

Given a trained ``Trainer`` (or any object exposing ``generate(c_ext)`` and a
``physics`` constraint set) and the data bundle, compute the full metric suite
and an efficiency summary.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..data.datasets import DataBundle
from ..models.forecaster import Forecaster, train_forecaster, evaluate_crps_per_cluster
from .metrics import (
    energy_score, mmd_rbf, tail_energy_score, feasibility_rate,
    downstream_crps_reduction,
)


def identify_hard_clusters(bundle: DataBundle, T: int, M: int,
                           epochs: int, device: str, frac: float = 0.34
                           ) -> List[int]:
    """Hard meteorological region = clusters in the top ``frac`` of baseline
    forecaster validation CRPS (where downstream error has most room to fall)."""
    f = Forecaster(T, M).to(device)
    f = train_forecaster(f, bundle.train.power.to(device), epochs=epochs, device=device)
    err = evaluate_crps_per_cluster(f, bundle.val.power, bundle.val.c_ext,
                                    bundle.n_clusters, device)
    ranked = sorted(err.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=-1)),
                    reverse=True)
    n_hard = max(1, int(round(frac * bundle.n_clusters)))
    return [c for c, _ in ranked[:n_hard]]


def synthesize_pool(model, bundle: DataBundle, n_total: int, device: str
                    ) -> torch.Tensor:
    """Generate a scenario pool whose cluster labels match the train mix."""
    labels = bundle.train.c_ext.numpy()
    n_clusters = model.n_clusters if hasattr(model, "n_clusters") else bundle.n_clusters
    # sample labels proportional to their training frequency (clip to model space)
    counts = np.array([(labels == c).sum() for c in range(n_clusters)], dtype=float)
    counts = counts / counts.sum() if counts.sum() > 0 else None
    draw = np.random.choice(n_clusters, size=n_total, p=counts)
    c_ext = torch.as_tensor(draw, dtype=torch.long, device=device)
    chunks = []
    bs = 256
    for i in range(0, n_total, bs):
        chunks.append(model.generate(c_ext[i:i + bs]).cpu())
    return torch.cat(chunks, dim=0)


def evaluate_model(model, bundle: DataBundle, cfg, seeds=(0, 1, 2),
                   verbose: bool = True) -> Dict[str, float]:
    device = cfg.device
    T, M = bundle.T, bundle.M
    real_test = bundle.test.power
    real_train = bundle.train.power.to(device)

    n_pool = min(len(bundle.train), 1024)
    gen = synthesize_pool(model, bundle, n_pool, device)

    # distributional + tail + feasibility metrics
    es = energy_score(real_test, gen)
    mmd = mmd_rbf(real_test, gen, tuple(cfg.eval.mmd_bandwidths))
    tail = tail_energy_score(real_test, gen, cfg.eval.tail_quantile)
    feas = feasibility_rate(gen.to(device), model.physics, cfg.eval.feasibility_tol)

    # downstream CRPS reduction (overall + hard region)
    hard = identify_hard_clusters(bundle, T, M, cfg.eval.forecaster_epochs, device)
    crps = downstream_crps_reduction(
        real_train, gen.to(device), real_test, bundle.test.c_ext,
        bundle.n_clusters, hard, T, M,
        epochs=cfg.eval.forecaster_epochs, device=device, seeds=seeds,
    )

    out = {
        "ES": es,
        "MMD": mmd,
        "Tail_ES": tail,
        "feasibility_rate": feas,
        "CRPS_reduction_overall": crps["overall"],
        "CRPS_reduction_hard": crps["hard_region"],
        "hard_clusters": hard,
    }
    if verbose:
        print("  ES={ES:.4f}  MMD={MMD:.4f}  Tail-ES={Tail_ES:.4f}  "
              "feas={feasibility_rate:.3f}  CRPS%(all)={CRPS_reduction_overall:.2f}  "
              "CRPS%(hard)={CRPS_reduction_hard:.2f}".format(**out))
    return out


def efficiency_summary(model, cfg) -> Dict[str, float]:
    """Parameter count and a single-sample inference-time estimate (Table 6)."""
    import time
    G = model.G
    n_params = sum(p.numel() for p in G.parameters()) / 1e6
    device = cfg.device
    c = torch.zeros(1, dtype=torch.long, device=device)
    # warmup + timed single-pass inference
    _ = model.generate(c)
    t0 = time.time()
    for _ in range(20):
        _ = model.generate(c)
    infer_ms = (time.time() - t0) / 20 * 1000.0
    return {"params_M": float(n_params), "infer_ms": float(infer_ms)}
