"""Structural-invariant loss L_struct.

Algorithm 1 lists ``L_str = patch_mask_recon(x_fake, real)`` and Section 3.3
states the structural term "preserves autocorrelation and trend continuity".
We therefore combine three structural invariants computed *between the batch of
generated trajectories and the batch of real trajectories* (distributional,
not paired):

  1. patch-mask reconstruction consistency -- the generated series, with random
     temporal patches masked and linearly in-filled, should remain close to the
     unmasked series (local smoothness / reconstructability invariant);
  2. autocorrelation matching -- the mean per-lag autocorrelation of generated
     series should match that of real series (preserves temporal correlation);
  3. trend continuity -- the distribution of first differences (mean abs.
     increment) of generated series should match real series.

L_struct is non-negative and, like L_phys, is gated by Delta-1.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _autocorr(x: torch.Tensor, lags: int) -> torch.Tensor:
    """Mean autocorrelation per lag, averaged over batch & channels.

    x: (B,T,M) -> (lags,)"""
    B, T, M = x.shape
    xc = x - x.mean(dim=1, keepdim=True)
    var = (xc ** 2).mean(dim=1, keepdim=True) + 1e-8
    acfs = []
    for l in range(1, lags + 1):
        num = (xc[:, l:, :] * xc[:, :-l, :]).mean(dim=1, keepdim=True)
        acfs.append((num / var).mean())                      # scalar
    return torch.stack(acfs)                                  # (lags,)


def _patch_mask_recon(x: torch.Tensor, patch_len: int, mask_ratio: float
                      ) -> torch.Tensor:
    """Mask random temporal patches, linearly interpolate across them, and
    measure reconstruction error against the original generated series."""
    B, T, M = x.shape
    n_patches = max(1, T // patch_len)
    n_mask = max(1, int(math.ceil(mask_ratio * n_patches)))  # manuscript: ceil(0.3*(T/4))
    recon = x.clone()
    loss = x.new_zeros(())
    for b in range(B):
        idx = torch.randperm(n_patches, device=x.device)[:n_mask]
        for pi in idx.tolist():
            s = pi * patch_len
            e = min(T, s + patch_len)
            if s == 0 or e >= T:
                # edge patch: hold-last / hold-first fill
                fill = x[b, max(0, s - 1), :] if s > 0 else x[b, min(T - 1, e), :]
                recon[b, s:e, :] = fill.unsqueeze(0)
            else:
                left = x[b, s - 1, :]
                right = x[b, e, :]
                steps = torch.linspace(0, 1, e - s + 2, device=x.device)[1:-1]
                recon[b, s:e, :] = (
                    left.unsqueeze(0) * (1 - steps).unsqueeze(1)
                    + right.unsqueeze(0) * steps.unsqueeze(1)
                )
    return F.mse_loss(recon, x)


def _patch_mask_recon_per_sample(x: torch.Tensor, patch_len: int,
                                 mask_ratio: float) -> torch.Tensor:
    """Per-sample (B,) version of the masked-reconstruction error."""
    B, T, M = x.shape
    n_patches = max(1, T // patch_len)
    n_mask = max(1, int(math.ceil(mask_ratio * n_patches)))  # manuscript: ceil(0.3*(T/4))
    recon = x.clone()
    for b in range(B):
        idx = torch.randperm(n_patches, device=x.device)[:n_mask]
        for pi in idx.tolist():
            s = pi * patch_len
            e = min(T, s + patch_len)
            if s == 0 or e >= T:
                fill = x[b, max(0, s - 1), :] if s > 0 else x[b, min(T - 1, e), :]
                recon[b, s:e, :] = fill.unsqueeze(0)
            else:
                left, right = x[b, s - 1, :], x[b, e, :]
                steps = torch.linspace(0, 1, e - s + 2, device=x.device)[1:-1]
                recon[b, s:e, :] = (left.unsqueeze(0) * (1 - steps).unsqueeze(1)
                                    + right.unsqueeze(0) * steps.unsqueeze(1))
    return ((recon - x) ** 2).mean(dim=(1, 2))                # (B,)


def _autocorr_per_sample(x: torch.Tensor, lags: int) -> torch.Tensor:
    """Per-sample autocorrelation, shape (B, lags)."""
    B, T, M = x.shape
    xc = x - x.mean(dim=1, keepdim=True)
    var = (xc ** 2).mean(dim=1, keepdim=True) + 1e-8
    acfs = []
    for l in range(1, lags + 1):
        num = (xc[:, l:, :] * xc[:, :-l, :]).mean(dim=1, keepdim=True)
        acfs.append((num / var).mean(dim=2).squeeze(1))       # (B,)
    return torch.stack(acfs, dim=1)                           # (B,lags)


class StructuralInvariants:
    def __init__(self, scfg):
        self.cfg = scfg

    def __call__(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """Distributional scalar L_struct (used by fixed-weight baselines)."""
        c = self.cfg
        l_recon = _patch_mask_recon(x_fake, c.patch_len, c.mask_ratio)
        acf_f = _autocorr(x_fake, c.acf_lags)
        acf_r = _autocorr(x_real, c.acf_lags).detach()
        l_acf = F.mse_loss(acf_f, acf_r)
        td_f = (x_fake[:, 1:, :] - x_fake[:, :-1, :]).abs().mean()
        td_r = (x_real[:, 1:, :] - x_real[:, :-1, :]).abs().mean().detach()
        l_trend = (td_f - td_r) ** 2
        return c.w_recon * l_recon + c.w_acf * l_acf + c.w_trend * l_trend

    def per_sample(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        """Per-sample (B,) L_struct so the per-sample gate g can weight it
        against per-sample L_phys, exactly as in Eq. (2)."""
        c = self.cfg
        l_recon = _patch_mask_recon_per_sample(x_fake, c.patch_len, c.mask_ratio)  # (B,)
        acf_f = _autocorr_per_sample(x_fake, c.acf_lags)                           # (B,lags)
        acf_r = _autocorr(x_real, c.acf_lags).detach()                            # (lags,)
        l_acf = ((acf_f - acf_r.unsqueeze(0)) ** 2).mean(dim=1)                    # (B,)
        td_f = (x_fake[:, 1:, :] - x_fake[:, :-1, :]).abs().mean(dim=(1, 2))       # (B,)
        td_r = (x_real[:, 1:, :] - x_real[:, :-1, :]).abs().mean().detach()
        l_trend = (td_f - td_r) ** 2                                              # (B,)
        return c.w_recon * l_recon + c.w_acf * l_acf + c.w_trend * l_trend        # (B,)
