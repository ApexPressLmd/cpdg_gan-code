"""A1 - Attention generator G(z, c).

Conditioning follows Section 3.2: the external label c_ext and the internal
controllable axes c_int are *concatenated, not summed*, so that (i) the
discriminator can attend to each independently and (ii) the orthogonality
regulariser of Section 3.6 acts on disjoint embedding blocks.  We therefore
expose ``embed_ext`` and ``embed_int`` separately.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention import AxialTransformer


class AttentionGenerator(nn.Module):
    def __init__(self, mcfg, T: int, M: int, n_clusters: int):
        super().__init__()
        self.T, self.M = T, M
        d = mcfg.attn_dim
        ce = mcfg.cond_embed_dim

        # disjoint conditioning blocks ---------------------------------------
        self.embed_ext = nn.Embedding(n_clusters, ce)          # c_ext -> R^ce
        self.embed_int = nn.Linear(mcfg.d_cint, ce)            # c_int -> R^ce

        # project [z, e_ext, e_int] into a (T, M, d) feature seed -------------
        in_dim = mcfg.d_z + 2 * ce
        self.to_seed = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Linear(d, T * M * (d // 2)),
        )
        self.seed_proj = nn.Linear(d // 2, d)

        self.backbone = AxialTransformer(
            dim=d, heads=mcfg.attn_heads, depth=mcfg.attn_depth, T=T, M=M,
            ff_mult=mcfg.ff_mult, dropout=mcfg.dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 1),
        )
        self.d_half = d // 2
        self.d = d

    def conditioning(self, c_ext: torch.Tensor, c_int: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two disjoint embedding blocks (e_ext, e_int)."""
        return self.embed_ext(c_ext), self.embed_int(c_int)

    def forward(self, z: torch.Tensor, c_ext: torch.Tensor, c_int: torch.Tensor
                ) -> torch.Tensor:
        B = z.shape[0]
        e_ext, e_int = self.conditioning(c_ext, c_int)
        cond = torch.cat([z, e_ext, e_int], dim=1)             # composite condition c
        seed = self.to_seed(cond).view(B, self.T, self.M, self.d_half)
        x = self.seed_proj(seed)                               # (B,T,M,d)
        x = self.backbone(x)
        out = self.head(x).squeeze(-1)                         # (B,T,M)
        # bounded, non-negative output via softplus-then-clip-friendly sigmoid scale
        return torch.sigmoid(out) * 1.05                       # matches capacity range
