"""Innovation 1 (Delta-1) - Diagnostic-Driven Physics-Gate.

Implements Equations (3) and (5) of Section 3.3 exactly:

    g     = sigma( W . [ r_phys.detach() , dist(c_int, mu) ] + b )      (3)
    L_reg = (1 - g) . L_phys + g . L_struct                             (5)

Design choices reproduced verbatim from Section 3.3:

* The physical residual entering the gate is **detached** so the gate learns to
  *read* the violation level as a diagnostic rather than trivially driving it to
  zero (the un-detached residual still flows through L_phys).
* The latent distance is a **robust standardised (Mahalanobis-style) norm**
  estimated from a **running covariance of c_int over the batch**, so extremity
  is comparable across axes of different scale.
* The gate is a **single affine map followed by a sigmoid** -- two parameter
  vectors only, no sub-network -- so the mechanism cannot smuggle in extra
  capacity that would explain the gains.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsGate(nn.Module):
    def __init__(self, d_cint: int, cov_ema: float = 0.99):
        super().__init__()
        # single affine map R^2 -> R^1  (W has shape (1,2), b scalar)
        self.W = nn.Parameter(torch.tensor([[1.0, 1.0]]))   # init: both diagnostics push g up
        self.b = nn.Parameter(torch.tensor(0.0))
        self.cov_ema = cov_ema
        # running statistics for the Mahalanobis distance of c_int
        self.register_buffer("running_mean", torch.zeros(d_cint))
        self.register_buffer("running_cov", torch.eye(d_cint))
        self.register_buffer("initialised", torch.tensor(0.0))

    # ---- running covariance update (EMA) ----------------------------------
    @torch.no_grad()
    def update_stats(self, c_int: torch.Tensor) -> None:
        mean = c_int.mean(dim=0)
        centred = c_int - mean
        cov = (centred.t() @ centred) / max(1, c_int.shape[0] - 1)
        if self.initialised.item() == 0.0:
            self.running_mean.copy_(mean)
            self.running_cov.copy_(cov + 1e-3 * torch.eye(cov.shape[0], device=cov.device))
            self.initialised.fill_(1.0)
        else:
            a = self.cov_ema
            self.running_mean.mul_(a).add_(mean, alpha=1 - a)
            self.running_cov.mul_(a).add_(cov, alpha=1 - a)

    def mahalanobis(self, c_int: torch.Tensor) -> torch.Tensor:
        """dist(c_int, mu): robust standardised norm, shape (B,)."""
        d = c_int.shape[1]
        cov = self.running_cov + 1e-3 * torch.eye(d, device=c_int.device)
        try:
            inv = torch.linalg.inv(cov)
        except RuntimeError:                      # singular -> diagonal fallback
            inv = torch.diag(1.0 / (torch.diagonal(cov) + 1e-3))
        centred = c_int - self.running_mean.unsqueeze(0)
        m2 = torch.einsum("bi,ij,bj->b", centred, inv, centred)
        return torch.sqrt(torch.clamp(m2, min=0.0) + 1e-8)

    # ---- the gate itself ---------------------------------------------------
    def forward(self, r_phys_detached: torch.Tensor, c_int: torch.Tensor
                ) -> torch.Tensor:
        """Return per-sample gate values g in (0,1), shape (B,)."""
        dist = self.mahalanobis(c_int)                          # (B,)
        feats = torch.stack([r_phys_detached, dist], dim=1)     # (B,2)
        logits = feats @ self.W.t() + self.b                    # (B,1)
        return torch.sigmoid(logits).squeeze(1)                 # (B,)

    def gated_regularizer(self, g: torch.Tensor, l_phys_vec: torch.Tensor,
                          l_struct_vec: torch.Tensor) -> torch.Tensor:
        """L_reg = (1-g)*L_phys + g*L_struct, averaged over the batch (Eq. 5)."""
        return ((1.0 - g) * l_phys_vec + g * l_struct_vec).mean()
