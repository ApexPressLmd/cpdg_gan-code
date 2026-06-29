"""PI-ST-GAN (P2) baseline - physics-informed spatio-temporal GAN.

Source atom A2 *in its unmixed form* (Section 2.3 / 4.2): the same axial-attention
spatio-temporal backbone, regularised by a differentiable feasibility penalty with
a **fixed** scalar weight - the very thing our Physics-Gate (Innovation 1) replaces
with a state-dependent gate.  No controllable axes, no structural-invariant routing,
no resampling.  This baseline is what isolates the value of *gating* the penalty:
removing only Delta1 from the full model and comparing against this fixed-weight
generator shows whether the gain comes from the gate rather than from the penalty
itself.  Independently tuned (Section 4.2): lr 1e-4, batch 64, 250 epochs,
fixed physics lambda.
"""
from __future__ import annotations

from typing import Optional

import torch

from src.models.generator import AttentionGenerator
from src.models.discriminator import AttentionDiscriminator
from src.models.physics import PhysicsConstraints
from .common import BaselineModel, train_wgan_gp
from .scengan import _ZeroCintGenerator


class PISTGANBaseline(BaselineModel):
    name = "PI-ST-GAN (P2)"

    def __init__(self, cfg, bundle, device: str = "cpu", phys_lambda: float = 1.0):
        self.device = device
        self.cfg = cfg
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        self.phys_lambda = phys_lambda
        T, M = bundle.T, bundle.M
        self.d_z = cfg.model.d_z
        gen = AttentionGenerator(cfg.model, T, M, self.n_clusters).to(device)
        self.G = _ZeroCintGenerator(gen, cfg.model.d_cint, self.d_z).to(device)
        self.D = AttentionDiscriminator(cfg.model, T, M, self.n_clusters).to(device)
        self._bundle = bundle

    def _fixed_physics(self, fake, real, c_ext):
        # uniform pressure regardless of where the sample lies (Section 2.3)
        return self.phys_lambda * self.physics.L_phys(fake)

    def fit(self, verbose: bool = False) -> "PISTGANBaseline":
        train_wgan_gp(
            self.G, self.D, self._bundle, self.cfg,
            lr=1e-4, epochs=min(250, getattr(self.cfg.train, "epochs", 250)),
            batch=64, n_critic=self.cfg.train.n_critic, gp_lambda=10.0,
            extra_g_loss=self._fixed_physics,
            device=self.device, verbose=verbose,
        )
        return self

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        self.G.eval()
        c_ext = c_ext.to(self.device)
        if n is not None:
            c_ext = c_ext[:1].repeat(n)
        z = torch.randn(c_ext.shape[0], self.d_z, device=self.device)
        x = self.G(z, c_ext)
        self.G.train()
        return x
