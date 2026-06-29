"""Innovation 2 (Delta-2) - Forecast-error-guided condition resampling.

Implements Equation (7) and the throttling/smoothing described in Section 3.4
and Algorithm 2:

    p(c_ext) = softmax( e(c_ext) / tau )                                  (7)

* recompute p only **every K epochs** (avoids chasing instantaneous error),
* apply an **exponential moving average** over successive distributions so the
  sampler *drifts* toward hard regions rather than oscillating,
* **floor** each cluster probability at a small epsilon to guarantee continued
  coverage of easy regions (prevents catastrophic forgetting),
* as tau -> infinity the distribution approaches uniform (recovering standard
  conditional generation); small tau greedily targets the worst region.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class ConditionSampler:
    """Maintains and updates the conditional sampling distribution p(c_ext)."""

    def __init__(self, n_clusters: int, tau: float, ema: float, floor: float,
                 device: str = "cpu"):
        self.n = n_clusters
        self.tau = tau
        self.ema = ema
        self.floor = floor
        self.device = device
        # start uniform (== standard conditional generation)
        self.p = np.full(n_clusters, 1.0 / n_clusters, dtype=np.float64)

    def _softmax(self, e: np.ndarray) -> np.ndarray:
        z = e / max(self.tau, 1e-8)
        z = z - np.nanmax(z)
        ez = np.exp(np.where(np.isnan(z), -np.inf, z))
        ez = np.where(np.isnan(ez), 0.0, ez)
        s = ez.sum()
        return ez / s if s > 0 else np.full(self.n, 1.0 / self.n)

    def update(self, errors: Dict[int, float]) -> np.ndarray:
        """Update p from per-cluster errors e(c_ext) (Eq. 7 + EMA + floor)."""
        e = np.array([errors.get(c, np.nan) for c in range(self.n)], dtype=np.float64)
        # clusters with no validation samples inherit the mean error
        if np.isnan(e).any():
            fill = np.nanmean(e) if not np.isnan(e).all() else 0.0
            e = np.where(np.isnan(e), fill, e)
        target = self._softmax(e)                          # Eq. (7)
        # EMA smoothing: p <- ema*p + (1-ema)*target
        self.p = self.ema * self.p + (1 - self.ema) * target
        # epsilon floor + renormalise
        self.p = np.maximum(self.p, self.floor)
        self.p = self.p / self.p.sum()
        return self.p.copy()

    def sample(self, n: int, generator: Optional[torch.Generator] = None
               ) -> torch.Tensor:
        """Draw n cluster labels c_ext ~ p."""
        idx = np.random.choice(self.n, size=n, p=self.p)
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    def state(self) -> Dict:
        return {"p": self.p.tolist(), "tau": self.tau, "ema": self.ema,
                "floor": self.floor}
