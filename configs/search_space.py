"""Hyper-parameter search space (Appendix A, Tables A1-A2).

Reproduces the grid over which the fusion model was tuned.  The chosen
*optimal* configuration (the defaults in :mod:`src.utils.config`) is recorded in
``OPTIMAL`` for reference.  ``iter_grid`` yields full overrides that can be
applied to a base :class:`~src.utils.config.Config` by ``scripts/hpo.py``.

Per Section 3.6 the loss weights are selected by grid search on validation CRPS;
the resampling-temperature sweep matches the grid plotted in Figure 7.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterator

# Search grid over architecture (Table A1: attn width/heads) and optimization /
# innovation (Table A2: loss weights, tau, K) hyperparameters. Each key is a
# (section, field) override path.
GRID: Dict[str, list] = {
    "train.lambda_reg":  [0.1, 0.5, 1.0, 2.0],
    "train.lambda_mi":   [0.01, 0.1, 0.5],
    "train.lambda_orth": [0.0, 0.1, 0.5],
    "train.tau":         [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0],  # Figure 7 sweep
    "train.K":           [5, 10, 20],
    "model.attn_heads":  [2, 4, 8],
    "model.attn_dim":    [64, 128, 256],
}

# Optimal configuration selected on validation CRPS (Section 3.6).
OPTIMAL: Dict[str, float] = {
    "train.lambda_reg":  1.0,
    "train.lambda_mi":   0.1,
    "train.lambda_orth": 0.1,
    "train.tau":         0.5,
    "train.K":           10,
    "model.attn_heads":  4,
    "model.attn_dim":    128,
}


def iter_grid(full_cartesian: bool = False) -> Iterator[Dict]:
    """Yield override dicts.

    ``full_cartesian=False`` (default) performs the coordinate-wise sweep used
    in the paper (vary one axis at a time around ``OPTIMAL``); setting it True
    yields the full Cartesian product (very large - for completeness only).
    """
    if full_cartesian:
        keys = list(GRID)
        for combo in itertools.product(*(GRID[k] for k in keys)):
            yield dict(zip(keys, combo))
        return
    yield dict(OPTIMAL)
    for key, values in GRID.items():
        for v in values:
            if v == OPTIMAL[key]:
                continue
            override = dict(OPTIMAL)
            override[key] = v
            yield override


def apply_override(cfg, override: Dict):
    """Apply a flat ``section.field -> value`` override dict onto a Config."""
    for path, value in override.items():
        section, field = path.split(".")
        setattr(getattr(cfg, section), field, value)
    return cfg
