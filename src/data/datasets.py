"""Datasets (Section 4.1, Table 2).

Three open benchmarks are *named* in the paper and the real loaders below point
at their public sources, with the DataCite-verified DOIs of Table 2:
the NREL WIND Toolkit (WTK, wind), the NSRDB+WIND U.S.-regions collection
(SWUS, solar), and the IEEE-DataPort hourly demand-weather dataset (HDW, load).
Because those archives are not bundled here, we additionally ship
physically-plausible **synthetic** generators that reproduce the salient
structure of each domain (diurnal solar envelope, autocorrelated wind with a
cubic speed->power map, temperature-driven load) together with meteorological
covariates that are genuinely causal for the output -- so that the A9 Granger
screen has real signal to retain and real nuisance covariates to discard.

Every loader returns a dict with:
    power : (N, T, M) float32   -- the multivariate power series x
    meteo : (N, T, P) float32   -- raw meteorological covariates w
plus metadata.  ``prepare_data`` then performs the chronological split,
per-channel normalisation and A9 conditioning, and wraps everything in a
``RenewableDataset``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .meteo_causal import MeteoCausalConditioner
from .preprocessing import ChannelNormalizer, chronological_split


# --------------------------------------------------------------------------- #
#  Synthetic generators                                                       #
# --------------------------------------------------------------------------- #
def _ar1(n: int, T: int, phi: float, sigma: float, rng) -> np.ndarray:
    """AR(1) latent series, shape (n, T)."""
    x = np.zeros((n, T))
    x[:, 0] = rng.normal(0, sigma, size=n)
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + rng.normal(0, sigma, size=n)
    return x


def make_synthetic_wind(n: int, T: int, M: int, P: int, seed: int = 0) -> Dict:
    """Multi-site wind: a shared regional wind-speed field + per-site noise,
    mapped through a clipped cubic turbine curve.  Causal meteo driver:
    wind speed (covariate 0) and its lag; nuisance: the remaining covariates."""
    rng = np.random.default_rng(seed)
    # regional wind speed (m/s-ish), autocorrelated + a slow diurnal drift
    base_speed = 8.0 + _ar1(n, T, phi=0.85, sigma=2.0, rng=rng)
    diurnal = 1.5 * np.sin(np.linspace(0, 2 * np.pi, T))[None, :]
    speed = np.clip(base_speed + diurnal, 0.0, None)        # (n, T)

    power = np.zeros((n, T, M))
    for m in range(M):
        site_speed = speed + _ar1(n, T, phi=0.6, sigma=0.8, rng=rng)  # site decorrelation
        site_speed = np.clip(site_speed, 0.0, None)
        # turbine power curve: cubic between cut-in (3) and rated (12), flat to cut-out (25)
        p = np.zeros_like(site_speed)
        ramp = (site_speed >= 3) & (site_speed < 12)
        rated = (site_speed >= 12) & (site_speed < 25)
        p[ramp] = ((site_speed[ramp] - 3) / (12 - 3)) ** 3
        p[rated] = 1.0
        power[:, :, m] = np.clip(p + rng.normal(0, 0.02, size=p.shape), 0, 1.05)

    meteo = np.zeros((n, T, P))
    meteo[:, :, 0] = speed / 25.0                            # causal: normalised wind speed
    meteo[:, :, 1] = np.roll(speed, 1, axis=1) / 25.0        # causal: lagged speed
    meteo[:, :, 2] = diurnal.repeat(n, 0)                    # weakly causal: time-of-day
    for j in range(3, P):                                    # nuisance covariates
        meteo[:, :, j] = _ar1(n, T, phi=0.5, sigma=1.0, rng=rng)
    return {"power": power.astype(np.float32), "meteo": meteo.astype(np.float32),
            "name": "synthetic_wind"}


def make_synthetic_solar(n: int, T: int, M: int, P: int, seed: int = 0) -> Dict:
    """Multi-site solar: a deterministic diurnal clear-sky envelope modulated by
    a stochastic cloud field.  Causal meteo driver: clear-sky index / irradiance;
    plus rapid cloud-ramp transients (the failure case of Figure 9)."""
    rng = np.random.default_rng(seed)
    hours = np.linspace(0, 24, T, endpoint=False)
    clear = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)   # daylight 6-18h
    clear = clear[None, :]                                       # (1, T)

    # cloud field: AR(1) clearness index in [0,1], occasionally a sharp ramp
    cloud = 0.5 + 0.5 * np.tanh(_ar1(n, T, phi=0.8, sigma=1.0, rng=rng))
    # inject rare rapid cloud-induced transients
    for i in range(n):
        if rng.random() < 0.15:
            t0 = rng.integers(int(0.3 * T), int(0.7 * T))
            cloud[i, t0:t0 + 2] *= rng.uniform(0.1, 0.4)

    power = np.zeros((n, T, M))
    for m in range(M):
        site_cloud = np.clip(cloud + _ar1(n, T, phi=0.5, sigma=0.3, rng=rng) * 0.1, 0, 1)
        p = clear * site_cloud
        power[:, :, m] = np.clip(p + rng.normal(0, 0.01, size=p.shape), 0, 1.05)

    meteo = np.zeros((n, T, P))
    meteo[:, :, 0] = (clear.repeat(n, 0) * cloud)               # causal: irradiance
    meteo[:, :, 1] = cloud                                      # causal: clearness index
    meteo[:, :, 2] = clear.repeat(n, 0)                         # causal: clear-sky envelope
    for j in range(3, P):
        meteo[:, :, j] = _ar1(n, T, phi=0.5, sigma=1.0, rng=rng)    # nuisance
    return {"power": power.astype(np.float32), "meteo": meteo.astype(np.float32),
            "name": "synthetic_solar"}


def make_synthetic_load(n: int, T: int, M: int, P: int, seed: int = 0) -> Dict:
    """Metropolitan demand-weather load (HDW domain): temperature-driven
    demand with a double-peak daily profile and co-located weather drivers.
    Causal meteo driver: temperature (and its square: cooling/heating)."""
    rng = np.random.default_rng(seed)
    hours = np.linspace(0, 24, T, endpoint=False)
    # double-peak daily load shape
    shape = (0.6 + 0.4 * np.exp(-((hours - 8) ** 2) / 6)
             + 0.5 * np.exp(-((hours - 19) ** 2) / 5))
    shape = shape / shape.max()
    shape = shape[None, :]

    temp = 15 + 10 * np.sin((hours - 9) / 24 * 2 * np.pi)[None, :] \
        + _ar1(n, T, phi=0.9, sigma=3.0, rng=rng)                  # deg C
    # U-shaped temperature response (heating + cooling)
    temp_resp = 0.02 * (temp - 18) ** 2
    base_load = shape * (0.7 + 0.3 * (temp_resp / (temp_resp.max() + 1e-8)))

    power = np.zeros((n, T, M))
    for m in range(M):
        load = base_load + _ar1(n, T, phi=0.7, sigma=0.05, rng=rng)
        power[:, :, m] = np.clip(load, 0, 1.05)

    meteo = np.zeros((n, T, P))
    meteo[:, :, 0] = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)  # causal: temp
    meteo[:, :, 1] = temp_resp / (temp_resp.max() + 1e-8)                    # causal: temp^2 resp
    meteo[:, :, 2] = shape.repeat(n, 0)                                      # causal: tod shape
    for j in range(3, P):
        meteo[:, :, j] = _ar1(n, T, phi=0.5, sigma=1.0, rng=rng)                # nuisance
    return {"power": power.astype(np.float32), "meteo": meteo.astype(np.float32),
            "name": "synthetic_load"}


SYNTHETIC = {
    "synthetic_wind": make_synthetic_wind,
    "synthetic_solar": make_synthetic_solar,
    "synthetic_load": make_synthetic_load,
}


# --------------------------------------------------------------------------- #
#  Real loaders (public sources from Table 2)                                 #
# --------------------------------------------------------------------------- #
# The three named benchmarks of Table 2, with the DataCite-verified DOIs used
# in the manuscript.  Keys are the dataset codes used throughout the paper.
_REAL_SOURCES = {
    "wtk": ("NREL WIND Toolkit (WTK) -- https://doi.org/10.7799/1329290 "
            "(landing: https://www.osti.gov/servlets/purl/1329290/)"),
    "swus": ("NSRDB+WIND U.S.-regions collection (SWUS) -- "
             "https://doi.org/10.17632/x6r9c6zvw6 "
             "(landing: https://data.mendeley.com/datasets/x6r9c6zvw6)"),
    "hdw": ("IEEE-DataPort hourly demand-weather dataset (HDW) -- "
            "https://doi.org/10.21227/fpqq-nr70 "
            "(landing: https://ieee-dataport.org/documents/"
            "hourly-electricity-demand-and-weather-data-major-us-cities-2019-2023)"),
}


def load_real(name: str, root: str, T: int, M: int, P: int) -> Dict:
    """Load a real benchmark from ``root`` if a prepared .npz is present.

    The expected file is ``{root}/{name}.npz`` containing arrays ``power``
    (N,T,M) and ``meteo`` (N,T,P).  If it is missing we raise a clear,
    actionable error pointing to the public source (Table 2 / Section 4.1).
    """
    path = os.path.join(root, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Real dataset '{name}' not found at {path}.\n"
            f"Download it from: {_REAL_SOURCES.get(name, '<see Table 2>')}\n"
            f"then save a prepared archive with arrays 'power' (N,T,M) and "
            f"'meteo' (N,T,P) to that path.  Until then use a synthetic dataset "
            f"(e.g. --dataset synthetic_wind) which reproduces the same protocol."
        )
    d = np.load(path)
    return {"power": d["power"].astype(np.float32),
            "meteo": d["meteo"].astype(np.float32), "name": name}


# --------------------------------------------------------------------------- #
#  Torch Dataset + full preparation pipeline                                  #
# --------------------------------------------------------------------------- #
class RenewableDataset(Dataset):
    """Holds normalised windows with their meteo covariates and c_ext labels."""

    def __init__(self, power: np.ndarray, meteo: np.ndarray, c_ext: np.ndarray):
        self.power = torch.from_numpy(power.astype(np.float32))
        self.meteo = torch.from_numpy(meteo.astype(np.float32))
        self.c_ext = torch.from_numpy(c_ext.astype(np.int64))

    def __len__(self) -> int:
        return self.power.shape[0]

    def __getitem__(self, i: int):
        return self.power[i], self.c_ext[i], self.meteo[i]


@dataclass
class DataBundle:
    train: RenewableDataset
    val: RenewableDataset
    test: RenewableDataset
    normalizer: ChannelNormalizer
    conditioner: MeteoCausalConditioner
    n_clusters: int
    T: int
    M: int
    P: int
    name: str

    def cluster_indices(self, split: str = "val") -> Dict[int, np.ndarray]:
        """Indices grouped by c_ext cluster, used by Delta-2 and hard-region eval."""
        ds = getattr(self, split)
        labels = ds.c_ext.numpy()
        return {c: np.where(labels == c)[0] for c in range(self.n_clusters)}


def prepare_data(cfg) -> DataBundle:
    """End-to-end data preparation following Section 4.1.

    cfg is the project ``Config``.
    """
    dcfg = cfg.data
    if dcfg.dataset in SYNTHETIC:
        raw = SYNTHETIC[dcfg.dataset](
            dcfg.n_samples, dcfg.horizon, dcfg.n_channels, dcfg.n_meteo, cfg.seed
        )
    else:
        raw = load_real(dcfg.dataset, dcfg.root, dcfg.horizon,
                        dcfg.n_channels, dcfg.n_meteo)

    power, meteo = raw["power"], raw["meteo"]
    n = power.shape[0]
    tr, va, te = chronological_split(n, tuple(dcfg.split))

    # per-channel normaliser fitted on TRAIN only (no leakage)
    norm = ChannelNormalizer(mode=dcfg.normalize).fit(power[tr])
    p_tr, p_va, p_te = norm.transform(power[tr]), norm.transform(power[va]), norm.transform(power[te])

    # A9 conditioner fitted on TRAIN only, applied everywhere
    cond = MeteoCausalConditioner(n_clusters=dcfg.n_clusters)
    c_tr = cond.fit_transform(meteo[tr], power[tr])
    c_va = cond.transform(meteo[va])
    c_te = cond.transform(meteo[te])

    return DataBundle(
        train=RenewableDataset(p_tr, meteo[tr], c_tr),
        val=RenewableDataset(p_va, meteo[va], c_va),
        test=RenewableDataset(p_te, meteo[te], c_te),
        normalizer=norm,
        conditioner=cond,
        n_clusters=dcfg.n_clusters,
        T=dcfg.horizon,
        M=dcfg.n_channels,
        P=dcfg.n_meteo,
        name=raw["name"],
    )
