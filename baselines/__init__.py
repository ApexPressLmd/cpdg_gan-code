"""Baseline models for the roster of Table 3 and the comparison of Table 4 (Sections 4.2-4.3).

Each baseline implements the uniform interface defined in
``baselines.common.BaselineModel`` and is constructed as
``Baseline(cfg, bundle, device=...)`` with a ``.fit()`` / ``.generate(c_ext)``
lifecycle, so the evaluation harness in :mod:`src.eval.evaluate` can score all
of them identically.

The ``REGISTRY`` maps the canonical paper names to constructors.  The
"fusion-without-Delta" comparison row is intentionally **not** a baseline here:
it is produced by the full :class:`src.training.trainer.Trainer` with
``AblationFlags.from_name("-delta1delta2")`` (see ``scripts/run_ablations.py``),
because it shares the fused architecture and only removes the two innovations.
"""
from __future__ import annotations

from .common import BaselineModel
from .wgan_gp import WGANGPBaseline
from .scengan import ScenGANBaseline
from .pistgan import PISTGANBaseline
from .vaegan import VAEGANBaseline
from .timegan import TimeGANBaseline
from .ts_diffusion import TSDiffusionBaseline

REGISTRY = {
    "wgan_gp": WGANGPBaseline,        # WGAN-GP
    "scengan": ScenGANBaseline,       # ScenGAN (P1)
    "pistgan": PISTGANBaseline,       # PI-ST-GAN (P2)
    "vaegan": VAEGANBaseline,         # VAE-GAN (P5)
    "timegan": TimeGANBaseline,       # TimeGAN
    "ts_diffusion": TSDiffusionBaseline,  # TS-Diffusion
}


def build_baseline(key: str, cfg, bundle, device: str = "cpu"):
    if key not in REGISTRY:
        raise KeyError(f"unknown baseline '{key}'. choices: {sorted(REGISTRY)}")
    return REGISTRY[key](cfg, bundle, device=device)


__all__ = [
    "BaselineModel", "REGISTRY", "build_baseline",
    "WGANGPBaseline", "ScenGANBaseline", "PISTGANBaseline",
    "VAEGANBaseline", "TimeGANBaseline", "TSDiffusionBaseline",
]
