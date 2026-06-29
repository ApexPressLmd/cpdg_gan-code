"""Common scaffolding for baseline models (Section 4.2).

Every baseline exposes the same minimal interface consumed by
``src.eval.evaluate.evaluate_model``:

    .generate(c_ext) -> (B, T, M) tensor
    .physics         -> PhysicsConstraints   (for the feasibility-rate metric)
    .n_clusters      -> int
    .G               -> the generator module  (for parameter counting, Table 6)

The unmixed source baselines are tuned independently (their own LRs / epochs,
Section 4.2) rather than inheriting the fusion pipeline's hyperparameters, as
demanded by Section 4.2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.physics import PhysicsConstraints
from src.training.wgan_gp import critic_loss, generator_adv_loss


class BaselineModel:
    """Mixin providing a uniform ``generate`` and book-keeping."""

    G: nn.Module
    n_clusters: int
    physics: PhysicsConstraints
    device: str
    d_z: int

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MLPConditionalGenerator(nn.Module):
    """A plain (non-attention) conditional generator used by WGAN-GP."""

    def __init__(self, d_z: int, n_clusters: int, T: int, M: int, hidden: int = 256,
                 cond_dim: int = 32):
        super().__init__()
        self.T, self.M = T, M
        self.d_z = d_z
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(d_z + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T * M),
        )

    def forward(self, z, c_ext):
        h = torch.cat([z, self.embed(c_ext)], dim=1)
        out = self.net(h).view(-1, self.T, self.M)
        return torch.sigmoid(out) * 1.05


class MLPConditionalCritic(nn.Module):
    def __init__(self, n_clusters: int, T: int, M: int, hidden: int = 256,
                 cond_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(T * M + cond_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, c_ext):
        h = torch.cat([x.reshape(x.shape[0], -1), self.embed(c_ext)], dim=1)
        return self.net(h).squeeze(1)


def sample_real_and_labels(bundle, n, device):
    labels = bundle.train.c_ext.numpy()
    idx = np.random.choice(len(labels), size=n)
    c = torch.as_tensor(labels[idx], dtype=torch.long, device=device)
    x = bundle.train.power[torch.as_tensor(idx)].to(device)
    return x, c


def train_wgan_gp(G, D, bundle, cfg, lr, epochs, batch, n_critic=5, gp_lambda=10.0,
                  extra_g_loss=None, device="cpu", verbose=False):
    """Generic conditional WGAN-GP training loop.

    ``extra_g_loss(x_fake, real, c_ext) -> scalar`` lets physics-/info-informed
    baselines add their own regulariser (e.g. PI-ST-GAN's fixed physics penalty).
    """
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    steps = max(1, len(bundle.train) // batch)
    for ep in range(epochs):
        for _ in range(steps):
            for _ in range(n_critic):
                real, c = sample_real_and_labels(bundle, batch, device)
                z = torch.randn(batch, G_z_dim(G), device=device)
                with torch.no_grad():
                    fake = G(z, c)
                d_loss = critic_loss(D, real, fake, c, gp_lambda)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            real, c = sample_real_and_labels(bundle, batch, device)
            z = torch.randn(batch, G_z_dim(G), device=device)
            fake = G(z, c)
            g_loss = generator_adv_loss(D, fake, c)
            if extra_g_loss is not None:
                g_loss = g_loss + extra_g_loss(fake, real, c)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        if verbose and ep % 50 == 0:
            print(f"    [baseline ep {ep}] d={float(d_loss):.3f} g={float(g_loss):.3f}")
    return G, D


def G_z_dim(G) -> int:
    """Best-effort latent-dim lookup for the generic loop."""
    if hasattr(G, "d_z"):
        return G.d_z
    # infer from first Linear in_features minus cond embedding
    for m in G.modules():
        if isinstance(m, nn.Linear):
            cond = G.embed.embedding_dim if hasattr(G, "embed") else 0
            return m.in_features - cond
    raise RuntimeError("cannot infer latent dim")
