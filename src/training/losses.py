"""Auxiliary losses.

Orthogonality regulariser (Section 3.6): "To prevent the two conditioning
channels from entangling, we add an orthogonality regularizer between their
embeddings."  We penalise the squared Frobenius norm of the batch
cross-covariance between the (disjoint) c_ext and c_int embedding blocks, which
drives the two channels to be linearly decorrelated.  Its strength
``lambda_orth`` is studied in Section 4.5 (Table A2 optimum = 0.1).
"""
from __future__ import annotations

import torch


def orthogonality_penalty(emb_ext: torch.Tensor, emb_int: torch.Tensor
                          ) -> torch.Tensor:
    """emb_ext: (B, ce), emb_int: (B, ce).  Returns a scalar penalty."""
    a = emb_ext - emb_ext.mean(dim=0, keepdim=True)
    b = emb_int - emb_int.mean(dim=0, keepdim=True)
    B = a.shape[0]
    cross = (a.t() @ b) / max(1, B - 1)              # (ce, ce) cross-covariance
    return (cross ** 2).sum()
