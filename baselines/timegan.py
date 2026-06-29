"""TimeGAN baseline (Yoon et al., 2019; Section 4.2).

A faithful conditional re-implementation of TimeGAN with its five networks -
embedder, recovery, sequence generator, supervisor, and discriminator - all
GRU-based and operating in a learned latent space.  Conditioning on the
external meteorological-regime label ``c_ext`` is injected by concatenating a
learned embedding to every input step (the projection trick is GAN-specific;
TimeGAN conditions through its inputs).

Training follows the original three-phase schedule:

    Phase 1  embedding network   :  minimise reconstruction  ||x - r(e(x))||^2
    Phase 2  supervised loss     :  minimise ||h_{t+1} - s(h_t)||^2 on real h
    Phase 3  joint training      :  alternate
                                       (a) generator   = adversarial
                                                       + eta * supervised
                                                       + moment matching
                                       (b) embedding    = reconstruction
                                                       + supervised
                                       (c) discriminator = real/fake/fake-e

The latent generator is driven by per-step noise; sampling produces a synthetic
latent path that the recovery network maps back to power space.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.physics import PhysicsConstraints
from .common import BaselineModel


class _CondGRU(nn.Module):
    """GRU stack that consumes [input, cond_embed] at every timestep."""

    def __init__(self, in_dim, hid, out_dim, n_clusters, cond_dim, layers=2,
                 out_act: Optional[str] = "sigmoid"):
        super().__init__()
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.gru = nn.GRU(in_dim + cond_dim, hid, num_layers=layers,
                          batch_first=True)
        self.head = nn.Linear(hid, out_dim)
        self.out_act = out_act

    def forward(self, x, c_ext):
        B, T, _ = x.shape
        e = self.embed(c_ext).unsqueeze(1).expand(B, T, -1)
        h, _ = self.gru(torch.cat([x, e], dim=-1))
        y = self.head(h)
        if self.out_act == "sigmoid":
            y = torch.sigmoid(y)
        return y


class _GFromRecovery(nn.Module):
    """Adapter exposing generator+recovery as G(z, c_ext) for the eval interface."""

    def __init__(self, parent: "TimeGANBaseline"):
        super().__init__()
        self.gen = parent.gen
        self.sup = parent.sup
        self.rec = parent.rec
        self.scale = parent.scale
        self.T, self.M, self.d_lat = parent.T, parent.M, parent.d_lat
        self.d_z = parent.d_lat

    def forward(self, z_seq, c_ext, c_int=None):
        # z_seq: (B, T, d_lat) latent noise path
        h = self.gen(z_seq, c_ext)
        h = self.sup(h, c_ext)
        x = self.rec(h, c_ext)
        return x * self.scale


class TimeGANBaseline(BaselineModel):
    name = "TimeGAN"

    def __init__(self, cfg, bundle, device: str = "cpu", d_lat: int = 24,
                 hid: int = 48, cond_dim: int = 16, eta: float = 10.0):
        self.cfg = cfg
        self._bundle = bundle
        self.device = device
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        self.T, self.M = bundle.T, bundle.M
        self.d_lat = d_lat
        self.eta = eta
        self.scale = 1.05  # capacity range, matches generator output bound
        nc, cd = self.n_clusters, cond_dim
        # five networks ------------------------------------------------------
        self.emb = _CondGRU(self.M, hid, d_lat, nc, cd)          # embedder
        self.rec = _CondGRU(d_lat, hid, self.M, nc, cd)          # recovery
        self.gen = _CondGRU(d_lat, hid, d_lat, nc, cd)           # generator
        self.sup = _CondGRU(d_lat, hid, d_lat, nc, cd)           # supervisor
        self.dis = _CondGRU(d_lat, hid, 1, nc, cd, out_act=None)  # discriminator
        for m in (self.emb, self.rec, self.gen, self.sup, self.dis):
            m.to(device)
        self.G = _GFromRecovery(self)

    # -- helpers -----------------------------------------------------------
    def _real_batch(self, n):
        idx = np.random.choice(len(self._bundle.train), size=n)
        x = self._bundle.train.power[torch.as_tensor(idx)].to(self.device)
        c = self._bundle.train.c_ext[torch.as_tensor(idx)].to(self.device).long()
        return x / self.scale, c  # normalise into roughly [0,1] for the AE

    def _noise(self, n):
        return torch.randn(n, self.T, self.d_lat, device=self.device)

    def fit(self, verbose: bool = False) -> "TimeGANBaseline":
        dev = self.device
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        batch = self.cfg.train.batch_size
        epochs = self.cfg.train.epochs
        lr = self.cfg.train.lr_g
        steps = max(1, len(self._bundle.train) // batch)

        opt_er = torch.optim.Adam(list(self.emb.parameters())
                                  + list(self.rec.parameters()), lr=lr)
        opt_gs = torch.optim.Adam(list(self.gen.parameters())
                                  + list(self.sup.parameters()), lr=lr)
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr)

        p1 = max(1, epochs // 4)   # embedding pretrain
        p2 = max(1, epochs // 4)   # supervised pretrain
        # ---- Phase 1: embedding network -----------------------------------
        for ep in range(p1):
            for _ in range(steps):
                x, c = self._real_batch(batch)
                h = self.emb(x, c)
                x_tilde = self.rec(h, c)
                loss = mse(x_tilde, x)
                opt_er.zero_grad(); loss.backward(); opt_er.step()
        # ---- Phase 2: supervised loss on real latents ---------------------
        for ep in range(p2):
            for _ in range(steps):
                x, c = self._real_batch(batch)
                with torch.no_grad():
                    h = self.emb(x, c)
                h_sup = self.sup(h, c)
                loss = mse(h_sup[:, :-1], h[:, 1:])
                opt_gs.zero_grad(); loss.backward(); opt_gs.step()
        # ---- Phase 3: joint adversarial training --------------------------
        for ep in range(epochs - p1 - p2):
            for _ in range(steps):
                # (a) generator + supervisor (twice, as in the original)
                for _ in range(2):
                    x, c = self._real_batch(batch)
                    z = self._noise(batch)
                    h = self.emb(x, c)
                    e_hat = self.gen(z, c)
                    h_hat = self.sup(e_hat, c)
                    h_sup = self.sup(h, c)
                    y_fake = self.dis(h_hat, c)
                    y_fake_e = self.dis(e_hat, c)
                    l_adv = bce(y_fake, torch.ones_like(y_fake)) \
                        + bce(y_fake_e, torch.ones_like(y_fake_e))
                    l_sup = mse(h_sup[:, :-1], h[:, 1:])
                    x_hat = self.rec(h_hat, c)
                    # moment matching on mean & std (original V1/V2 terms)
                    l_mom = (torch.abs(x_hat.mean((0, 1)) - x.mean((0, 1))).mean()
                             + torch.abs(x_hat.std((0, 1)) - x.std((0, 1))).mean())
                    g_loss = l_adv + self.eta * l_sup + l_mom
                    opt_gs.zero_grad(); g_loss.backward(); opt_gs.step()
                    # (b) embedding network (reconstruction + supervised)
                    x, c = self._real_batch(batch)
                    h = self.emb(x, c)
                    x_tilde = self.rec(h, c)
                    h_sup = self.sup(h, c)
                    e_loss = 10.0 * mse(x_tilde, x) + 0.1 * mse(h_sup[:, :-1], h[:, 1:])
                    opt_er.zero_grad(); e_loss.backward(); opt_er.step()
                # (c) discriminator
                x, c = self._real_batch(batch)
                z = self._noise(batch)
                with torch.no_grad():
                    h = self.emb(x, c)
                    e_hat = self.gen(z, c)
                    h_hat = self.sup(e_hat, c)
                y_real = self.dis(h, c)
                y_fake = self.dis(h_hat, c)
                y_fake_e = self.dis(e_hat, c)
                d_loss = (bce(y_real, torch.ones_like(y_real))
                          + bce(y_fake, torch.zeros_like(y_fake))
                          + bce(y_fake_e, torch.zeros_like(y_fake_e)))
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            if verbose and ep % 50 == 0:
                print(f"    [TimeGAN ep {ep}] d={float(d_loss):.3f} g={float(g_loss):.3f}")
        return self

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        for m in (self.gen, self.sup, self.rec):
            m.eval()
        c_ext = c_ext.to(self.device).long()
        B = c_ext.shape[0]
        z = self._noise(B)
        e_hat = self.gen(z, c_ext)
        h_hat = self.sup(e_hat, c_ext)
        x = self.rec(h_hat, c_ext) * self.scale
        for m in (self.gen, self.sup, self.rec):
            m.train()
        return x
