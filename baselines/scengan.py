"""ScenGAN (P1) baseline - attention-intensive generator, plain WGAN-GP.

This is source-atom A1 *in its unmixed form* (Section 2.2 / 4.2): the same axial
attention generator and discriminator that our full model uses, but trained with
a bare Wasserstein-GP objective - no controllable axes (A6), no meteo-causal
conditioning beyond the c_ext label it is given, no physics (A2), and neither
innovation.  c_int is held at zero so the attention backbone is exercised purely
as a conditional scenario generator.  Independently tuned (Section 4.2):
lr 2e-4, batch 64, 250 epochs, attn heads = 4.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.generator import AttentionGenerator
from src.models.discriminator import AttentionDiscriminator
from src.models.physics import PhysicsConstraints
from .common import BaselineModel, train_wgan_gp


class _ZeroCintGenerator(nn.Module):
    """Adapter exposing forward(z, c_ext) by feeding a zero c_int to A1's G."""

    def __init__(self, gen: AttentionGenerator, d_cint: int, d_z: int):
        super().__init__()
        self.gen = gen
        self.d_cint = d_cint
        self.d_z = d_z
        # expose the embedding the generic loop's G_z_dim fallback may probe
        self.embed = gen.embed_ext

    def forward(self, z, c_ext):
        c_int = torch.zeros(z.shape[0], self.d_cint, device=z.device)
        return self.gen(z, c_ext, c_int)


class ScenGANBaseline(BaselineModel):
    name = "ScenGAN (P1)"

    def __init__(self, cfg, bundle, device: str = "cpu"):
        self.device = device
        self.cfg = cfg
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        T, M = bundle.T, bundle.M
        self.d_z = cfg.model.d_z
        gen = AttentionGenerator(cfg.model, T, M, self.n_clusters).to(device)
        self.G = _ZeroCintGenerator(gen, cfg.model.d_cint, self.d_z).to(device)
        self.D = AttentionDiscriminator(cfg.model, T, M, self.n_clusters).to(device)
        self._bundle = bundle

    def fit(self, verbose: bool = False) -> "ScenGANBaseline":
        train_wgan_gp(
            self.G, self.D, self._bundle, self.cfg,
            lr=2e-4, epochs=min(250, getattr(self.cfg.train, "epochs", 250)),
            batch=64, n_critic=self.cfg.train.n_critic, gp_lambda=10.0,
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
