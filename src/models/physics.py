"""A2 - Differentiable physical-feasibility penalty.

Section 3.1 / 3.3 / Algorithm 1:

    r_phys = relu(phys_violation(x_fake))   # ramp / bounds / nonneg
    L_phys = mean(r_phys)                    # only active constraints contribute

The residual ``r_phys`` is rectified so that *only active constraints*
contribute, exactly as stated in Section 3.3.  A per-sample scalar
(``residual_per_sample``) is what the Physics-Gate consumes as a diagnostic
(detached); the full element-wise residual is what L_phys minimises.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class PhysicsConstraints:
    """Encodes ramp-rate, capacity-bound and non-negativity constraints."""

    def __init__(self, pcfg):
        self.ramp_max = pcfg.ramp_max
        self.capacity = pcfg.capacity
        self.nonneg = pcfg.nonneg
        self.reduce = pcfg.reduce

    # -- element-wise rectified violations ----------------------------------
    def violations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return rectified violation tensors. x: (B,T,M)."""
        out = {}
        # ramp-rate: |x_t - x_{t-1}| <= ramp_max
        dx = x[:, 1:, :] - x[:, :-1, :]
        out["ramp"] = F.relu(dx.abs() - self.ramp_max)         # (B,T-1,M)
        # capacity upper bound: x <= capacity
        out["capacity"] = F.relu(x - self.capacity)            # (B,T,M)
        # non-negativity: x >= 0  -> violation = relu(-x)
        if self.nonneg:
            out["nonneg"] = F.relu(-x)                          # (B,T,M)
        return out

    def residual_elementwise(self, x: torch.Tensor) -> torch.Tensor:
        """Total rectified residual reduced over the (T,M) grid per sample-term,
        returned at full element granularity then summed across constraints.

        We return a (B,) tensor: the per-sample aggregate of all active
        constraint violations (mean over time/sites, summed over constraint
        types).  This is r_phys aggregated per sample."""
        v = self.violations(x)
        per_sample = 0.0
        for key, t in v.items():
            per_sample = per_sample + t.mean(dim=tuple(range(1, t.dim())))
        return per_sample                                       # (B,)

    def L_phys(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar feasibility loss = mean over batch of per-sample residual."""
        return self.residual_elementwise(x).mean()

    def residual_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample scalar diagnostic r_phys (B,) for the Physics-Gate."""
        return self.residual_elementwise(x)

    def feasibility_mask(self, x: torch.Tensor, tol: float) -> torch.Tensor:
        """Boolean (B,): True if a sample has total violation <= tol
        (used for the feasibility-rate metric in Section 4.1)."""
        return self.residual_elementwise(x) <= tol
