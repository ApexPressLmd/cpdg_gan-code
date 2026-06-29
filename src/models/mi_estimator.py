"""A6 - Mutual-information controllable latent axes.

Section 3.6: "The internal axes c_int are learned by maximizing the mutual
information between the axes and generated outputs under spectral
normalization."

We use the InfoGAN variational lower bound: an auxiliary recognition network Q,
sharing an attention encoder, predicts the posterior over c_int from the
generated trajectory.  For continuous axes c_int we model a diagonal Gaussian
posterior and maximise its log-likelihood of the true latent code, which is a
tractable lower bound on I(c_int; x_hat).  Spectral normalisation is applied to
Q's linear layers (as specified) to stabilise the estimator.

The negative bound ``L_mi`` is what the generator objective adds (Algorithm 1:
``L_mi = -I(c_int, x_fake)``).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from .attention import AxialTransformer


def _sn(linear: nn.Linear, enabled: bool) -> nn.Module:
    return spectral_norm(linear) if enabled else linear


class MIEstimator(nn.Module):
    """Recognition network Q(c_int | x_hat) with Gaussian posterior."""

    def __init__(self, mcfg, T: int, M: int):
        super().__init__()
        d = mcfg.attn_dim
        sn = mcfg.spectral_norm_mi
        self.in_proj = _sn(nn.Linear(1, d), sn)
        self.backbone = AxialTransformer(
            dim=d, heads=mcfg.attn_heads, depth=max(1, mcfg.attn_depth - 1),
            T=T, M=M, ff_mult=mcfg.ff_mult, dropout=mcfg.dropout,
        )
        self.norm = nn.LayerNorm(d)
        self.mu = _sn(nn.Linear(d, mcfg.d_cint), sn)
        self.log_sigma = _sn(nn.Linear(d, mcfg.d_cint), sn)

    def posterior(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x.unsqueeze(-1))
        h = self.backbone(h)
        pooled = self.norm(h).mean(dim=(1, 2))
        mu = self.mu(pooled)
        log_sigma = self.log_sigma(pooled).clamp(-5.0, 2.0)
        return mu, log_sigma

    def mutual_information(self, x_fake: torch.Tensor, c_int: torch.Tensor
                           ) -> torch.Tensor:
        """Variational lower bound I_LB(c_int; x_fake) (a scalar, to be
        maximised).  Equivalent to the expected Gaussian log-likelihood of the
        true code under the predicted posterior, up to the constant entropy of
        the (fixed) code prior, which does not affect the generator gradient."""
        mu, log_sigma = self.posterior(x_fake)
        # Gaussian log-likelihood of true c_int under N(mu, sigma^2)
        ll = -0.5 * (
            ((c_int - mu) ** 2) / (torch.exp(2 * log_sigma) + 1e-8)
            + 2 * log_sigma
            + torch.log(torch.tensor(2 * torch.pi))
        )
        return ll.sum(dim=1).mean()
