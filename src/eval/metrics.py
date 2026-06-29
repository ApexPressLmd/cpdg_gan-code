"""Evaluation metrics (Section 4.1).

Reported quantities:
  * Energy Score (ES)          -- distributional fidelity (lower better)
  * Maximum Mean Discrepancy   -- multi-bandwidth RBF MMD (lower better)
  * Tail-ES                    -- ES restricted to the top-5% ramp events
  * Physical-feasibility rate  -- fraction of generated samples that satisfy
                                  all constraints within tolerance
  * Downstream CRPS reduction  -- % improvement of a forecaster trained on
                                  real+synthetic vs. real-only (overall + hard
                                  meteorological region), via the A5 forecaster.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..models.physics import PhysicsConstraints
from ..models.forecaster import Forecaster, train_forecaster, evaluate_crps


# --------------------------------------------------------------------------- #
#  Energy Score                                                               #
# --------------------------------------------------------------------------- #
def _pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Euclidean distances between rows of a (n,d) and b (m,d) -> (n,m)."""
    return torch.cdist(a, b, p=2)


def energy_score(real: torch.Tensor, gen: torch.Tensor,
                 max_real: int = 512, max_gen: int = 256) -> float:
    """Mean Energy Score of the generated ensemble against real observations.

    ES = mean_y [ E_X ||X - y|| - 0.5 E_{X,X'} ||X - X'|| ],
    with X, X' independent draws from the generator and y ranging over real
    samples.  Lower is better.  Series are flattened to vectors."""
    real = real.reshape(real.shape[0], -1)
    gen = gen.reshape(gen.shape[0], -1)
    if real.shape[0] > max_real:
        idx = torch.randperm(real.shape[0])[:max_real]
        real = real[idx]
    if gen.shape[0] > max_gen:
        idx = torch.randperm(gen.shape[0])[:max_gen]
        gen = gen[idx]
    term1 = _pairwise_dist(gen, real).mean()                  # E||X - y||
    term2 = _pairwise_dist(gen, gen).mean()                   # E||X - X'||
    return float((term1 - 0.5 * term2).item())


# --------------------------------------------------------------------------- #
#  Maximum Mean Discrepancy (multi-bandwidth RBF)                             #
# --------------------------------------------------------------------------- #
def mmd_rbf(real: torch.Tensor, gen: torch.Tensor,
            bandwidths: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
            max_n: int = 512) -> float:
    real = real.reshape(real.shape[0], -1)
    gen = gen.reshape(gen.shape[0], -1)
    if real.shape[0] > max_n:
        real = real[torch.randperm(real.shape[0])[:max_n]]
    if gen.shape[0] > max_n:
        gen = gen[torch.randperm(gen.shape[0])[:max_n]]

    def _kernel(a, b):
        d2 = torch.cdist(a, b, p=2) ** 2
        # median heuristic scale times each bandwidth multiplier
        med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0)
        k = 0.0
        for bw in bandwidths:
            k = k + torch.exp(-d2 / (2 * (bw ** 2) * med + 1e-8))
        return k / len(bandwidths)

    kxx = _kernel(real, real).mean()
    kyy = _kernel(gen, gen).mean()
    kxy = _kernel(real, gen).mean()
    return float((kxx + kyy - 2 * kxy).clamp_min(0.0).item())


# --------------------------------------------------------------------------- #
#  Tail-ES (top-5% ramp events)                                               #
# --------------------------------------------------------------------------- #
def ramp_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-sample maximum absolute first difference (a ramp severity score)."""
    dx = (x[:, 1:, :] - x[:, :-1, :]).abs()
    return dx.amax(dim=(1, 2))


def tail_energy_score(real: torch.Tensor, gen: torch.Tensor,
                      quantile: float = 0.95) -> float:
    """ES computed on the top (1-quantile) ramp-event subset of BOTH sets."""
    r_ramp = ramp_magnitude(real)
    g_ramp = ramp_magnitude(gen)
    r_thr = torch.quantile(r_ramp, quantile)
    g_thr = torch.quantile(g_ramp, quantile)
    real_tail = real[r_ramp >= r_thr]
    gen_tail = gen[g_ramp >= g_thr]
    if real_tail.shape[0] < 2 or gen_tail.shape[0] < 2:
        return float("nan")
    return energy_score(real_tail, gen_tail)


# --------------------------------------------------------------------------- #
#  Physical-feasibility rate                                                  #
# --------------------------------------------------------------------------- #
def feasibility_rate(gen: torch.Tensor, physics: PhysicsConstraints,
                     tol: float = 1e-4) -> float:
    mask = physics.feasibility_mask(gen, tol)
    return float(mask.float().mean().item())


# --------------------------------------------------------------------------- #
#  Downstream CRPS reduction (overall + hard region)                          #
# --------------------------------------------------------------------------- #
def downstream_crps_reduction(
    real_train: torch.Tensor,
    synth: torch.Tensor,
    real_test: torch.Tensor,
    test_labels: torch.Tensor,
    n_clusters: int,
    hard_clusters: List[int],
    T: int,
    M: int,
    epochs: int = 60,
    device: str = "cpu",
    seeds: Tuple[int, ...] = (0,),
) -> Dict[str, float]:
    """% CRPS reduction of real+synth vs real-only on the test split.

    Returns {'overall': .., 'hard_region': ..}.  Averaged over ``seeds`` to
    reduce forecaster-init noise."""
    overall, hard = [], []
    hard_mask = np.isin(test_labels.cpu().numpy(), hard_clusters)
    test_hard = real_test[torch.as_tensor(hard_mask)]
    for s in seeds:
        torch.manual_seed(s)
        f_base = Forecaster(T, M).to(device)
        f_base = train_forecaster(f_base, real_train, epochs=epochs, device=device)
        torch.manual_seed(s)
        f_aug = Forecaster(T, M).to(device)
        f_aug = train_forecaster(
            f_aug, torch.cat([real_train, synth], dim=0), epochs=epochs, device=device
        )
        c_base = evaluate_crps(f_base, real_test, device)
        c_aug = evaluate_crps(f_aug, real_test, device)
        overall.append(100.0 * (c_base - c_aug) / (c_base + 1e-12))
        if test_hard.shape[0] > 1:
            ch_base = evaluate_crps(f_base, test_hard, device)
            ch_aug = evaluate_crps(f_aug, test_hard, device)
            hard.append(100.0 * (ch_base - ch_aug) / (ch_base + 1e-12))
    return {
        "overall": float(np.mean(overall)),
        "hard_region": float(np.mean(hard)) if hard else float("nan"),
    }
