"""TS-Diffusion baseline (Section 4.2).

A conditional denoising-diffusion probabilistic model (DDPM, Ho et al., 2020)
over fixed-length multi-site power sequences, conditioned on the external
meteorological-regime label ``c_ext``.  This is the modern non-adversarial
generative baseline.  The denoiser is a residual temporal network that takes
the noisy sequence, a sinusoidal embedding of the diffusion timestep, and the
conditioning embedding, and predicts the injected noise (the standard
epsilon-prediction parameterisation).  Sampling runs the ancestral reverse
chain from Gaussian noise.

Exposes ``.G`` (the denoiser) for parameter counting and the uniform
``.generate(c_ext)`` interface used by the evaluation harness.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.models.physics import PhysicsConstraints
from .common import BaselineModel


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device) / max(1, half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


class _ResBlock(nn.Module):
    def __init__(self, ch, cond_ch):
        super().__init__()
        self.fc1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.fc2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.cond = nn.Linear(cond_ch, ch)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.fc1(self.act(self.norm1(x)))
        h = h + self.cond(cond).unsqueeze(-1)
        h = self.fc2(self.act(self.norm2(h)))
        return x + h


class _Denoiser(nn.Module):
    """epsilon_theta(x_t, t, c_ext) over (B, T, M) sequences (channels = M)."""

    def __init__(self, T, M, n_clusters, ch=64, t_dim=64, cond_dim=32, blocks=4):
        super().__init__()
        self.T, self.M = T, M
        self.t_dim = t_dim
        self.in_proj = nn.Conv1d(M, ch, 1)
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.embed = nn.Embedding(n_clusters, cond_dim)
        self.c_mlp = nn.Sequential(nn.Linear(cond_dim, ch), nn.SiLU(), nn.Linear(ch, ch))
        self.blocks = nn.ModuleList([_ResBlock(ch, ch) for _ in range(blocks)])
        self.out = nn.Conv1d(ch, M, 1)
        self.d_z = T * M  # nominal latent size for book-keeping

    def forward(self, x, t, c_ext):
        # x: (B, T, M) -> conv over time with channels=M
        h = self.in_proj(x.transpose(1, 2))               # (B, ch, T)
        cond = self.t_mlp(_timestep_embedding(t, self.t_dim)) + self.c_mlp(self.embed(c_ext))
        for blk in self.blocks:
            h = blk(h, cond)
        eps = self.out(h).transpose(1, 2)                 # (B, T, M)
        return eps


class TSDiffusionBaseline(BaselineModel):
    name = "TS-Diffusion"

    def __init__(self, cfg, bundle, device: str = "cpu", n_steps: int = 100,
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.cfg = cfg
        self._bundle = bundle
        self.device = device
        self.n_clusters = bundle.n_clusters
        self.physics = PhysicsConstraints(cfg.physics)
        self.T, self.M = bundle.T, bundle.M
        self.scale = 1.05
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.G = _Denoiser(self.T, self.M, self.n_clusters).to(device)
        self.d_z = self.G.d_z

    def _real_batch(self, n):
        idx = np.random.choice(len(self._bundle.train), size=n)
        x = self._bundle.train.power[torch.as_tensor(idx)].to(self.device)
        c = self._bundle.train.c_ext[torch.as_tensor(idx)].to(self.device).long()
        # map roughly to [-1, 1] for the diffusion process
        return (x / self.scale) * 2.0 - 1.0, c

    def fit(self, verbose: bool = False) -> "TSDiffusionBaseline":
        opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg.train.lr_g,
                               betas=(0.9, 0.999))
        batch = self.cfg.train.batch_size
        steps = max(1, len(self._bundle.train) // batch)
        for ep in range(self.cfg.train.epochs):
            for _ in range(steps):
                x0, c = self._real_batch(batch)
                t = torch.randint(0, self.n_steps, (batch,), device=self.device)
                noise = torch.randn_like(x0)
                ab = self.alpha_bar[t].view(-1, 1, 1)
                x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
                eps = self.G(x_t, t, c)
                loss = ((eps - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if verbose and ep % 50 == 0:
                print(f"    [TS-Diffusion ep {ep}] mse={float(loss):.4f}")
        return self

    @torch.no_grad()
    def generate(self, c_ext: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        self.G.eval()
        c_ext = c_ext.to(self.device).long()
        B = c_ext.shape[0]
        x = torch.randn(B, self.T, self.M, device=self.device)
        for i in reversed(range(self.n_steps)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps = self.G(x, t, c_ext)
            alpha = self.alphas[i]
            ab = self.alpha_bar[i]
            coef = (1 - alpha) / (1 - ab).sqrt()
            mean = (x - coef * eps) / alpha.sqrt()
            if i > 0:
                noise = torch.randn_like(x)
                sigma = self.betas[i].sqrt()
                x = mean + sigma * noise
            else:
                x = mean
        # map back from [-1,1] to power range and clamp to physical bounds
        x = (x + 1.0) * 0.5 * self.scale
        x = x.clamp(min=0.0, max=self.scale)
        self.G.train()
        return x
