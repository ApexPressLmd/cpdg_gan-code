"""WGAN-GP scenario-generation baseline (Table 3 roster, Section 4.2).

Model-free conditional WGAN-GP with a plain (non-attention) MLP generator and
projection-conditioned critic.  This is the classical renewable-scenario GAN of
Section 2.1; it has neither attention, controllable axes, physics, nor any of
the two innovations.  Trained under its *own* independently tuned settings (lr 1e-4,
batch 64, 200 epochs, GP lambda 10) per the independent-tuning rule of
Section 4.2.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.models.physics import PhysicsConstraints
from .common import (
    BaselineModel,
    MLPConditionalGenerator,
    MLPConditionalCritic,
    train_wgan_gp,
)


class WGANGPBaseline(BaselineModel):
    name = "WGAN-GP"

    def __init__(self, cfg, bundle, device: str = "cpu"):
        self.device = device
        self.cfg = cfg
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        T, M = bundle.T, bundle.M
        self.d_z = cfg.model.d_z
        self.G = MLPConditionalGenerator(self.d_z, self.n_clusters, T, M).to(device)
        self.D = MLPConditionalCritic(self.n_clusters, T, M).to(device)
        self._bundle = bundle

    # -- independently tuned (Section 4.2): lr 1e-4, batch 64, 200 epochs, GP lambda=10
    def fit(self, verbose: bool = False) -> "WGANGPBaseline":
        train_wgan_gp(
            self.G, self.D, self._bundle, self.cfg,
            lr=1e-4, epochs=self._epochs(200), batch=64,
            n_critic=self.cfg.train.n_critic, gp_lambda=10.0,
            device=self.device, verbose=verbose,
        )
        return self

    def _epochs(self, default: int) -> int:
        # honour the smoke-config short schedule when present
        return min(default, getattr(self.cfg.train, "epochs", default))

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        self.G.eval()
        c_ext = c_ext.to(self.device)
        B = c_ext.shape[0] if n is None else n
        if n is not None:
            c_ext = c_ext[:1].repeat(n)
        z = torch.randn(B, self.d_z, device=self.device)
        x = self.G(z, c_ext)
        self.G.train()
        return x
