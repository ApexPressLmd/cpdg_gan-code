"""Attention discriminator / critic D(x, c_ext).

A WGAN critic with the same axial-attention backbone as the generator, made
conditional via the projection-discriminator trick
(Miyato & Koyama, 2018): the scalar critic value is the unconditional score
plus an inner product between the pooled feature and a learned c_ext embedding.
No output non-linearity (Wasserstein critic).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AxialTransformer


class AttentionDiscriminator(nn.Module):
    def __init__(self, mcfg, T: int, M: int, n_clusters: int):
        super().__init__()
        d = mcfg.attn_dim
        self.in_proj = nn.Linear(1, d)
        self.backbone = AxialTransformer(
            dim=d, heads=mcfg.attn_heads, depth=mcfg.attn_depth, T=T, M=M,
            ff_mult=mcfg.ff_mult, dropout=mcfg.dropout,
        )
        self.norm = nn.LayerNorm(d)
        self.score = nn.Linear(d, 1)                  # unconditional critic head
        self.embed_ext = nn.Embedding(n_clusters, d)  # projection conditioning

    def forward(self, x: torch.Tensor, c_ext: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M)
        h = self.in_proj(x.unsqueeze(-1))             # (B,T,M,d)
        h = self.backbone(h)
        pooled = self.norm(h).mean(dim=(1, 2))        # (B,d) global pool
        base = self.score(pooled).squeeze(-1)         # (B,)
        proj = (pooled * self.embed_ext(c_ext)).sum(dim=1)  # projection term
        return base + proj
