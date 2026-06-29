"""Preprocessing utilities (Section 4.1).

* chronological 70/15/15 split (no leakage across windows),
* per-channel normalisation,
* daily windowing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ChannelNormalizer:
    """Per-channel normaliser. ``mode`` in {per_channel_minmax, per_channel_std}.

    Fitted on the training split only and reused for val/test (no leakage).
    """

    mode: str = "per_channel_minmax"
    a_: np.ndarray = None    # min  or mean
    b_: np.ndarray = None    # max  or std

    def fit(self, x: np.ndarray) -> "ChannelNormalizer":
        # x: (N, T, M)
        flat = x.reshape(-1, x.shape[-1])
        if self.mode == "per_channel_minmax":
            self.a_ = flat.min(axis=0)
            self.b_ = flat.max(axis=0)
            self.b_ = np.where(self.b_ - self.a_ < 1e-8, self.a_ + 1.0, self.b_)
        elif self.mode == "per_channel_std":
            self.a_ = flat.mean(axis=0)
            self.b_ = flat.std(axis=0)
            self.b_ = np.where(self.b_ < 1e-8, 1.0, self.b_)
        else:
            raise ValueError(self.mode)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "per_channel_minmax":
            return (x - self.a_) / (self.b_ - self.a_)
        return (x - self.a_) / self.b_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "per_channel_minmax":
            return x * (self.b_ - self.a_) + self.a_
        return x * self.b_ + self.a_


def chronological_split(
    n: int, split: Tuple[float, float, float]
) -> Tuple[slice, slice, slice]:
    """Return contiguous (train, val, test) slices preserving time order."""
    assert abs(sum(split) - 1.0) < 1e-6
    n_tr = int(round(split[0] * n))
    n_va = int(round(split[1] * n))
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)
