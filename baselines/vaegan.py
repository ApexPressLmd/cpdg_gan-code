"""VAEGAN (P5) baseline - controllable VAE-GAN with mutual information.

Source atom A6 *in its unmixed form* (Section 2.2 / 4.2): a variational
auto-encoder whose decoder is also trained adversarially, with a controllable
latent block recovered by an InfoGAN-style recognition network (mutual-information
maximisation under spectral normalisation).  It has controllability but neither
physics (A2), meteo-causal *screening* beyond the c_ext label, nor either
innovation.  Comparing this against the full model isolates the value of the
couplings rather than of controllability per se.

Architecture (Section 4.2 - only the data interface is adapted):
  Encoder  E(x, c_ext)            -> q(u | x, c_ext),  u = [z ; c_int]
  Decoder  G(u, c_ext)            -> x_hat                         (= ``self.G``)
  Critic   D(x, c_ext)            -> Wasserstein score
  Recog.   Q(c_int | x_hat)       -> reused src.models.MIEstimator
Losses: reconstruction (MSE) + KL(q||N(0,I)) + WGAN-GP adversarial + lambda_mi * (-I).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.physics import PhysicsConstraints
from src.models.mi_estimator import MIEstimator
from src.models.discriminator import AttentionDiscriminator
from src.training.wgan_gp import critic_loss, generator_adv_loss
from .common import BaselineModel, sample_real_and_labels


class _Encoder(nn.Module):
    def __init__(self, n_clusters, T, M, d_u, hidden=256, cond_dim=32):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(T * M + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, d_u)
        self.log_var = nn.Linear(hidden, d_u)

    def forward(self, x, c_ext):
        h = torch.cat([x.reshape(x.shape[0], -1), self.embed(c_ext)], dim=1)
        h = self.net(h)
        return self.mu(h), self.log_var(h).clamp(-6.0, 4.0)


class _Decoder(nn.Module):
    """Generator/decoder G([z, c_int], c_ext) -> x_hat. Exposed as self.G."""

    def __init__(self, d_u, n_clusters, T, M, hidden=256, cond_dim=32):
        super().__init__()
        self.T, self.M, self.d_u = T, M, d_u
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.net = nn.Sequential(
            nn.Linear(d_u + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, T * M),
        )

    def forward(self, u, c_ext):
        h = torch.cat([u, self.embed(c_ext)], dim=1)
        return torch.sigmoid(self.net(h)).view(-1, self.T, self.M) * 1.05


class VAEGANBaseline(BaselineModel):
    name = "VAEGAN (P5)"

    def __init__(self, cfg, bundle, device: str = "cpu"):
        self.device = device
        self.cfg = cfg
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        T, M = bundle.T, bundle.M
        self.T, self.M = T, M
        self.d_z = cfg.model.d_z
        self.d_cint = cfg.model.d_cint
        self.d_u = self.d_z + self.d_cint
        self.E = _Encoder(self.n_clusters, T, M, self.d_u).to(device)
        self.G = _Decoder(self.d_u, self.n_clusters, T, M).to(device)
        self.D = AttentionDiscriminator(cfg.model, T, M, self.n_clusters).to(device)
        self.Q = MIEstimator(cfg.model, T, M).to(device)
        self._bundle = bundle

    def _split(self, u):
        return u[:, :self.d_z], u[:, self.d_z:]   # z, c_int

    def fit(self, verbose: bool = False) -> "VAEGANBaseline":
        cfg = self.cfg
        epochs = min(250, getattr(cfg.train, "epochs", 250))
        batch = 64
        lam_mi = cfg.train.lambda_mi
        opt_g = torch.optim.Adam(
            list(self.G.parameters()) + list(self.E.parameters())
            + list(self.Q.parameters()), lr=2e-4, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.9))
        steps = max(1, len(self._bundle.train) // batch)
        for ep in range(epochs):
            for _ in range(steps):
                # ---- critic updates on prior samples -----------------------
                for _ in range(cfg.train.n_critic):
                    real, c = sample_real_and_labels(self._bundle, batch, self.device)
                    u = torch.randn(batch, self.d_u, device=self.device)
                    with torch.no_grad():
                        fake = self.G(u, c)
                    d_loss = critic_loss(self.D, real, fake, c, cfg.train.gp_lambda)
                    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                # ---- VAE + adversarial + MI generator update ----------------
                real, c = sample_real_and_labels(self._bundle, batch, self.device)
                mu, log_var = self.E(real, c)
                std = torch.exp(0.5 * log_var)
                u_enc = mu + std * torch.randn_like(std)
                rec = self.G(u_enc, c)
                l_rec = ((rec - real) ** 2).mean()
                l_kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).mean()
                # prior path for adversarial + MI
                u_prior = torch.randn(batch, self.d_u, device=self.device)
                _, c_int_prior = self._split(u_prior)
                gen = self.G(u_prior, c)
                l_adv = generator_adv_loss(self.D, gen, c)
                l_mi = -self.Q.mutual_information(gen, c_int_prior)
                l_g = l_adv + l_rec + cfg.train.vae_beta * l_kl + lam_mi * l_mi
                opt_g.zero_grad(); l_g.backward(); opt_g.step()
            if verbose and ep % 50 == 0:
                print(f"    [VAEGAN ep {ep}] adv={float(l_adv):.3f} "
                      f"rec={float(l_rec):.3f} kl={float(l_kl):.3f}")
        return self

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        self.G.eval()
        c_ext = c_ext.to(self.device)
        if n is not None:
            c_ext = c_ext[:1].repeat(n)
        u = torch.randn(c_ext.shape[0], self.d_u, device=self.device)
        x = self.G(u, c_ext)
        self.G.train()
        return x
