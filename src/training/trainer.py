"""Trainer - faithful implementation of Algorithms 1 and 2.

The inner loop is Algorithm 1 (training step with the diagnostic-driven
Physics-Gate); the outer loop is Algorithm 2 (forecast-error-guided resampling,
fired every K epochs).  Every component the paper ablates is switchable through
``AblationFlags`` so that ``scripts/run_ablations.py`` can reproduce Table 5:

    Full | -Delta2 | -Delta1 | -Delta1Delta2 (== fusion-no-delta) | -A6 | -A9
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import warnings
import numpy as np
import torch
import torch.nn as nn

from ..data.datasets import DataBundle
from ..models.generator import AttentionGenerator
from ..models.discriminator import AttentionDiscriminator
from ..models.mi_estimator import MIEstimator
from ..models.physics import PhysicsConstraints
from ..models.structural import StructuralInvariants
from ..models.physics_gate import PhysicsGate
from ..models.forecaster import (
    Forecaster, train_forecaster, evaluate_crps_per_cluster,
)
from .wgan_gp import critic_loss, generator_adv_loss
from .losses import orthogonality_penalty
from .resampling import ConditionSampler


@dataclass
class AblationFlags:
    physics_gate: bool = True       # Delta 1
    resampling: bool = True         # Delta 2
    mi: bool = True                 # A6 mutual-information regulariser
    controllable_axes: bool = True  # A6 internal axes c_int (non-trivial)
    meteo_cond: bool = True         # A9 meteorological conditioning

    @classmethod
    def from_name(cls, name: str) -> "AblationFlags":
        name = name.lower()
        f = cls()
        if name in ("full", "ours"):
            return f
        if name in ("-delta2", "no_resampling"):
            f.resampling = False
        elif name in ("-delta1", "no_gate"):
            f.physics_gate = False
        elif name in ("-delta1delta2", "fusion_no_delta", "fusion", "no_delta"):
            f.physics_gate = False
            f.resampling = False
        elif name in ("-a6", "no_controllable"):
            f.controllable_axes = False
            f.mi = False
        elif name in ("-a9", "no_meteo"):
            f.meteo_cond = False
            f.resampling = False   # resampling loses the cluster structure it needs
        else:
            raise ValueError(f"unknown ablation '{name}'")
        return f


class ClusterRealSampler:
    """Draws real windows conditioned on a desired c_ext label vector."""

    def __init__(self, bundle: DataBundle, device: str):
        self.power = bundle.train.power.to(device)
        self.labels = bundle.train.c_ext.numpy()
        self.n_clusters = bundle.n_clusters
        self.by_cluster = {c: np.where(self.labels == c)[0]
                           for c in range(self.n_clusters)}
        # fall back to all indices for empty clusters
        all_idx = np.arange(len(self.labels))
        for c in range(self.n_clusters):
            if len(self.by_cluster[c]) == 0:
                self.by_cluster[c] = all_idx
        self.device = device

    def sample(self, c_ext: torch.Tensor) -> torch.Tensor:
        idx = []
        for c in c_ext.cpu().numpy():
            pool = self.by_cluster[int(c)]
            idx.append(np.random.choice(pool))
        return self.power[torch.as_tensor(idx, device=self.device)]

    def sample_uniform(self, n: int) -> (torch.Tensor, torch.Tensor):
        idx = np.random.choice(len(self.labels), size=n)
        c = torch.as_tensor(self.labels[idx], dtype=torch.long, device=self.device)
        return self.power[torch.as_tensor(idx, device=self.device)], c


class Trainer:
    def __init__(self, cfg, bundle: DataBundle, flags: Optional[AblationFlags] = None):
        self.cfg = cfg
        self.bundle = bundle
        self.flags = flags or AblationFlags()
        self.device = cfg.device
        T, M = bundle.T, bundle.M
        # under -A9, collapse the conditioning to a single trivial cluster
        self.n_clusters = bundle.n_clusters if self.flags.meteo_cond else 1

        mcfg = cfg.model
        self.G = AttentionGenerator(mcfg, T, M, self.n_clusters).to(self.device)
        self.D = AttentionDiscriminator(mcfg, T, M, self.n_clusters).to(self.device)
        self.Q = MIEstimator(mcfg, T, M).to(self.device)
        self.gate = PhysicsGate(mcfg.d_cint, cfg.train.gate_cov_ema).to(self.device)

        self.physics = PhysicsConstraints(cfg.physics)
        self.structural = StructuralInvariants(cfg.structural)

        tc = cfg.train
        g_params = list(self.G.parameters()) + list(self.Q.parameters()) \
            + list(self.gate.parameters())
        self.opt_g = torch.optim.Adam(g_params, lr=tc.lr_g, betas=tc.betas)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=tc.lr_d, betas=tc.betas)

        self.real_sampler = ClusterRealSampler(bundle, self.device)
        self.cond_sampler = ConditionSampler(
            self.n_clusters, tc.tau, tc.resample_ema, tc.resample_floor, self.device
        )
        # persistent downstream forecaster (Section 3.5: retrained in outer loop)
        self.forecaster: Optional[Forecaster] = None
        self.history: List[Dict] = []
        self.p_history: List[List[float]] = []

    # ------------------------------------------------------------------ utils
    def sample_cint(self, B: int) -> torch.Tensor:
        if self.flags.controllable_axes:
            return torch.randn(B, self.cfg.model.d_cint, device=self.device)
        return torch.zeros(B, self.cfg.model.d_cint, device=self.device)

    def sample_cext(self, B: int) -> torch.Tensor:
        if self.flags.meteo_cond and self.flags.resampling:
            return self.cond_sampler.sample(B)            # Delta-2 biased sampling
        if self.flags.meteo_cond:
            _, c = self.real_sampler.sample_uniform(B)    # data-distribution clusters
            return c
        return torch.zeros(B, dtype=torch.long, device=self.device)  # -A9

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        self.G.eval()
        if n is not None:
            c_ext = c_ext.repeat(n) if c_ext.numel() == 1 else c_ext[:n]
        B = c_ext.shape[0]
        z = torch.randn(B, self.cfg.model.d_z, device=self.device)
        c_int = self.sample_cint(B)
        x = self.G(z, c_ext, c_int)
        self.G.train()
        return x

    # --------------------------------------------------------- inner step (Alg.1)
    def train_step(self) -> Dict[str, float]:
        tc = self.cfg.train
        B = tc.batch_size

        # ----- critic updates (n_critic) ------------------------------------
        d_loss_val = 0.0
        for _ in range(tc.n_critic):
            c_ext = self.sample_cext(B)
            real = self.real_sampler.sample(c_ext) if self.flags.meteo_cond \
                else self.real_sampler.sample_uniform(B)[0]
            z = torch.randn(B, self.cfg.model.d_z, device=self.device)
            c_int = self.sample_cint(B)
            with torch.no_grad():
                fake = self.G(z, c_ext, c_int)
            d_loss = critic_loss(self.D, real, fake, c_ext, tc.gp_lambda)
            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            d_loss_val = float(d_loss.item())

        # ----- generator update --------------------------------------------
        c_ext = self.sample_cext(B)
        real = self.real_sampler.sample(c_ext) if self.flags.meteo_cond \
            else self.real_sampler.sample_uniform(B)[0]
        z = torch.randn(B, self.cfg.model.d_z, device=self.device)
        c_int = self.sample_cint(B)                       # c_int ~ axes()
        x_fake = self.G(z, c_ext, c_int)                  # A1 attention generator

        L_adv = generator_adv_loss(self.D, x_fake, c_ext)

        # A2 physics: per-sample rectified residual + scalar feasibility loss
        r_phys = self.physics.residual_per_sample(x_fake)         # (B,)
        l_phys_vec = r_phys                                       # L_phys = mean(r_phys)
        # structural invariants (per-sample for the gate)
        l_struct_vec = self.structural.per_sample(x_fake, real)   # (B,)

        # ---- Delta-1 gated regulariser vs. fixed-weight fallback -----------
        if self.flags.physics_gate:
            self.gate.update_stats(c_int.detach())
            g = self.gate(r_phys.detach(), c_int)                 # Eq. (3)
            L_reg = self.gate.gated_regularizer(g, l_phys_vec, l_struct_vec)  # Eq. (5)
            g_mean = float(g.mean().item())
        else:
            # fixed equal-weight combination (the standard physics-informed
            # baseline that Delta-1 replaces): lambda=0.5 on each term
            L_reg = 0.5 * l_phys_vec.mean() + 0.5 * l_struct_vec.mean()
            g_mean = float("nan")

        # A6 mutual-information regulariser
        if self.flags.mi:
            L_mi = -self.Q.mutual_information(x_fake, c_int)       # = -I(c_int, x)
        else:
            L_mi = x_fake.new_zeros(())

        # orthogonality between disjoint conditioning embedding blocks
        if self.flags.meteo_cond:
            e_ext, e_int = self.G.conditioning(c_ext, c_int)
            L_orth = orthogonality_penalty(e_ext, e_int)
        else:
            L_orth = x_fake.new_zeros(())

        L_G = (L_adv
               + tc.lambda_reg * L_reg
               + tc.lambda_mi * L_mi
               + tc.lambda_orth * L_orth)

        self.opt_g.zero_grad()
        L_G.backward()
        if tc.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.G.parameters(), tc.grad_clip)
        self.opt_g.step()

        return {
            "d_loss": d_loss_val,
            "g_loss": float(L_G.item()),
            "adv": float(L_adv.item()),
            "L_phys": float(l_phys_vec.mean().item()),
            "L_struct": float(l_struct_vec.mean().item()),
            "L_reg": float(L_reg.item()),
            "L_mi": float(L_mi.item()),
            "L_orth": float(L_orth.item()),
            "gate_g": g_mean,
        }

    # --------------------------------------------------- outer loop (Alg. 2)
    def outer_resample_step(self) -> Dict[int, float]:
        """Every K epochs: synthesise per cluster, retrain the forecaster on
        real+synth, evaluate per-cluster validation error, update p(c_ext)."""
        bundle = self.bundle
        T, M = bundle.T, bundle.M
        real_train = bundle.train.power.to(self.device)
        synth_list = [real_train]
        N = self.cfg.train.resample_synth_per_cluster
        for c in range(self.n_clusters):
            c_vec = torch.full((N,), c, dtype=torch.long, device=self.device)
            synth_list.append(self.generate(c_vec))
        aug = torch.cat(synth_list, dim=0)

        # (re)train the lightweight forecaster on real + synthetic
        fc = Forecaster(T, M).to(self.device)
        fc = train_forecaster(fc, aug, epochs=self.cfg.eval.forecaster_epochs,
                              device=self.device)
        self.forecaster = fc

        # per-cluster validation error e(c_ext)
        errors = evaluate_crps_per_cluster(
            fc, bundle.val.power, bundle.val.c_ext, self.n_clusters, self.device
        )
        if self.flags.resampling:
            p = self.cond_sampler.update(errors)          # Eq. (7) + EMA + floor
            self.p_history.append(p.tolist())
        return errors

    # ------------------------------------------------------------------ fit
    def fit(self, verbose: bool = True) -> "Trainer":
        tc = self.cfg.train
        bundle = self.bundle
        steps_per_epoch = max(1, len(bundle.train) // tc.batch_size)
        for epoch in range(tc.epochs):
            ep_logs = []
            for _ in range(steps_per_epoch):
                ep_logs.append(self.train_step())
            # outer Delta-2 loop every K epochs (and once at the very end)
            if (epoch + 1) % tc.K == 0:
                self.outer_resample_step()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                agg = {k: float(np.nanmean([d[k] for d in ep_logs])) for k in ep_logs[0]}
            agg["epoch"] = epoch
            self.history.append(agg)
            if verbose and (epoch % tc.log_every == 0 or epoch == tc.epochs - 1):
                print(f"[epoch {epoch:4d}] d={agg['d_loss']:+.3f} "
                      f"g={agg['g_loss']:+.3f} adv={agg['adv']:+.3f} "
                      f"Lphys={agg['L_phys']:.4f} Lstruct={agg['L_struct']:.4f} "
                      f"g_gate={agg['gate_g']:.3f}")
        # make sure a forecaster exists for downstream eval even if K>epochs
        if self.forecaster is None:
            self.outer_resample_step()
        return self

    # ------------------------------------------------------------- checkpoint
    def state_dict(self) -> Dict:
        return {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "Q": self.Q.state_dict(),
            "gate": self.gate.state_dict(),
            "p": self.cond_sampler.state(),
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
