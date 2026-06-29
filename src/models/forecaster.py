"""A5 - Downstream forecaster.

A lightweight probabilistic forecaster that predicts the tail of each daily
window (the last ``horizon_out`` steps) from its head, for every channel.  It
outputs a per-step Gaussian, which admits a closed-form CRPS (Gneiting &
Raftery, 2007) used both as the training objective surrogate and as the
augmentation-gain metric.

Per Section 3.5 the forecaster is "kept lightweight and frozen during inner-loop
generator updates; it is retrained only inside the outer loop", which prevents
it from chasing a moving generator and keeps the Delta-2 error signal well
defined.

The augmentation gain reported in Tables 4-5 is the % CRPS reduction of a
forecaster trained on real+synthetic vs. one trained on real alone, evaluated on
the *real* validation/test split.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SQRT_PI = math.sqrt(math.pi)


def gaussian_crps(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor
                  ) -> torch.Tensor:
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma^2) against truth y.

    CRPS = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ], z=(y-mu)/sigma.
    Returns the elementwise CRPS (same shape as inputs)."""
    sigma = sigma.clamp_min(1e-6)
    z = (y - mu) / sigma
    normal = torch.distributions.Normal(0.0, 1.0)
    cdf = normal.cdf(z)
    pdf = torch.exp(normal.log_prob(z))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1.0 / _SQRT_PI)


class Forecaster(nn.Module):
    """GRU encoder over the history + Gaussian heads over the future."""

    def __init__(self, T: int, M: int, horizon_out: Optional[int] = None,
                 hidden: int = 64, layers: int = 1):
        super().__init__()
        self.T = T
        self.M = M
        self.horizon_out = horizon_out or max(1, T // 2)
        self.horizon_in = T - self.horizon_out
        self.gru = nn.GRU(M, hidden, num_layers=layers, batch_first=True)
        self.mu_head = nn.Linear(hidden, self.horizon_out * M)
        self.logsig_head = nn.Linear(hidden, self.horizon_out * M)

    def forward(self, hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hist: (B, horizon_in, M)
        _, h = self.gru(hist)
        h = h[-1]                                              # (B,hidden)
        B = h.shape[0]
        mu = self.mu_head(h).view(B, self.horizon_out, self.M)
        sigma = torch.nn.functional.softplus(
            self.logsig_head(h).view(B, self.horizon_out, self.M)) + 1e-3
        return mu, sigma

    def split(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x[:, :self.horizon_in, :], x[:, self.horizon_in:, :]


def train_forecaster(
    model: Forecaster,
    x_train: torch.Tensor,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Forecaster:
    """Fit the forecaster on a set of full windows by minimising mean CRPS."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(x_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            hist, fut = model.split(xb)
            mu, sigma = model(hist)
            loss = gaussian_crps(mu, sigma, fut).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_crps(model: Forecaster, x_eval: torch.Tensor, device: str = "cpu"
                  ) -> float:
    model.eval()
    x_eval = x_eval.to(device)
    hist, fut = model.split(x_eval)
    mu, sigma = model(hist)
    return float(gaussian_crps(mu, sigma, fut).mean().item())


@torch.no_grad()
def evaluate_crps_per_cluster(
    model: Forecaster,
    x_eval: torch.Tensor,
    c_eval: torch.Tensor,
    n_clusters: int,
    device: str = "cpu",
) -> Dict[int, float]:
    """Per-cluster validation CRPS e(c_ext) consumed by Delta-2 (Algorithm 2)."""
    model.eval()
    x_eval = x_eval.to(device)
    hist, fut = model.split(x_eval)
    mu, sigma = model(hist)
    crps = gaussian_crps(mu, sigma, fut).mean(dim=(1, 2)).cpu().numpy()  # (N,)
    c = c_eval.cpu().numpy()
    out = {}
    for k in range(n_clusters):
        m = c == k
        out[k] = float(crps[m].mean()) if m.any() else float("nan")
    return out
