"""A1 - Attention building blocks.

Figure 1 specifies the generator as "a stack of self-attention blocks over
both the temporal and the site axes".  We implement *axial* attention: each
block attends along the time axis (for every site independently) and then along
the site axis (for every time step independently).  This captures long-range
temporal dependencies *and* cross-site correlation with O(T^2 M + T M^2) cost,
matching the complexity stated in Section 3.2.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn

# Force the math Scaled-Dot-Product-Attention backend.  The default fused /
# flash CPU kernel selected by ``nn.MultiheadAttention`` (when need_weights=
# False) has no implemented backward on CPU-only PyTorch builds, so we pin the
# numerically-equivalent math kernel, which trains identically and portably.
try:  # PyTorch >= 2.1
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def _math_sdpa():
        return sdpa_kernel(SDPBackend.MATH)
except Exception:  # pragma: no cover - older fallbacks
    try:
        def _math_sdpa():
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True
            )
    except Exception:
        def _math_sdpa():
            return contextlib.nullcontext()


class AxialAttentionBlock(nn.Module):
    """One transformer block applying temporal- then site-axis attention.

    Input / output shape: (B, T, M, d).
    """

    def __init__(self, dim: int, heads: int, ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        self.attn_m = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, M, D = x.shape

        # ---- temporal attention: sequences of length T, batch B*M ----------
        xt = x.permute(0, 2, 1, 3).reshape(B * M, T, D)      # (B*M, T, D)
        h = self.norm_t(xt)
        with _math_sdpa():
            a, _ = self.attn_t(h, h, h, need_weights=True, average_attn_weights=True)
        xt = xt + a
        x = xt.reshape(B, M, T, D).permute(0, 2, 1, 3)        # (B, T, M, D)

        # ---- site attention: sequences of length M, batch B*T ---------------
        xm = x.reshape(B * T, M, D)                           # (B*T, M, D)
        h = self.norm_m(xm)
        with _math_sdpa():
            a, _ = self.attn_m(h, h, h, need_weights=False)
        xm = xm + a
        x = xm.reshape(B, T, M, D)

        # ---- position-wise feed-forward -------------------------------------
        x = x + self.ff(self.norm_ff(x))
        return x


class AxialTransformer(nn.Module):
    """A stack of ``depth`` axial blocks with learnable T and M positional
    embeddings (added once at the input)."""

    def __init__(self, dim: int, heads: int, depth: int, T: int, M: int,
                 ff_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.pos_t = nn.Parameter(torch.randn(1, T, 1, dim) * 0.02)
        self.pos_m = nn.Parameter(torch.randn(1, 1, M, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [AxialAttentionBlock(dim, heads, ff_mult, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_t + self.pos_m
        for blk in self.blocks:
            x = blk(x)
        return x
