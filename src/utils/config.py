"""Configuration objects.

Default values reproduce the *optimal configuration* of Appendix A (Tables A1 and A2)
(the full model used in Tables 4-6):

    lambda_reg  = 1.0     (gated regularizer weight)
    lambda_mi   = 0.1     (mutual-information regularizer)
    lambda_orth = 0.1     (orthogonality between c_ext and c_int)
    tau         = 0.5     (Delta-2 resampling temperature)
    K           = 10      (outer-loop period, in epochs)
    attn heads  = 4 ; attn dim = 128

The hyperparameter *search ranges* themselves live in
``configs/search_space.py`` so ``scripts/hpo.py`` can reproduce the grid.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "synthetic_wind"          # synthetic_wind|synthetic_solar|synthetic_load|wtk|swus|hdw
    root: str = "./data"
    horizon: int = 24                        # T: daily window (hourly)
    n_channels: int = 4                       # M: number of sites / co-located channels
    n_meteo: int = 8                          # raw meteorological covariates w
    split: tuple = (0.70, 0.15, 0.15)         # chronological 70/15/15 (Section 4.1)
    n_clusters: int = 6                       # number of c_ext meteorological clusters
    n_samples: int = 4096                     # synthetic-data size (real loaders override)
    normalize: str = "per_channel_minmax"     # per-channel normalisation (Section 4.1)


@dataclass
class ModelConfig:
    d_z: int = 32                             # latent noise dim
    d_cint: int = 4                           # controllable internal axes c_int (continuous)
    attn_dim: int = 128                       # transformer width (Table A1, architecture)
    attn_heads: int = 4                       # attention heads (Table A1, architecture)
    attn_depth: int = 3                       # number of axial self-attention blocks (A1)
    ff_mult: int = 2
    dropout: float = 0.0
    cond_embed_dim: int = 64                  # shared embedding dim for c_ext / c_int blocks
    spectral_norm_mi: bool = True             # A6 uses spectral normalisation


@dataclass
class PhysicsConfig:
    ramp_max: float = 0.30                    # max |x_t - x_{t-1}| in normalised units
    capacity: float = 1.05                    # capacity bound (slightly above 1 after norm)
    nonneg: bool = True                       # enforce x >= 0
    reduce: str = "mean_time"                 # per-sample scalar aggregation for r_phys


@dataclass
class StructuralConfig:
    patch_len: int = 4                        # patch length for patch-mask reconstruction
    mask_ratio: float = 0.30                  # fraction of patches masked
    w_recon: float = 1.0
    w_acf: float = 0.5                        # autocorrelation-matching weight
    w_trend: float = 0.5                      # trend-continuity (first-difference) weight
    acf_lags: int = 6


@dataclass
class TrainConfig:
    epochs: int = 250                         # Table A2 (optimization)
    batch_size: int = 64                      # Table A2 (optimization)
    lr_g: float = 2e-4                        # Table A2 (optimization)
    lr_d: float = 2e-4
    betas: tuple = (0.5, 0.9)                 # standard WGAN-GP Adam betas
    n_critic: int = 5                         # discriminator updates per generator update
    gp_lambda: float = 10.0                   # gradient-penalty coefficient
    lambda_reg: float = 1.0                   # gated regulariser weight  (Table A2)
    lambda_mi: float = 0.1                    # MI regulariser            (Table A2)
    lambda_orth: float = 0.1                  # orthogonality penalty     (Table A2)
    vae_beta: float = 1.0                     # KL weight for the VAE-GAN baseline (P5)
    # Delta-2 outer loop:
    use_resampling: bool = True               # Delta 2 enabled
    K: int = 10                               # recompute p(c_ext) every K epochs (Table A2)
    tau: float = 0.5                          # resampling temperature     (Table A2)
    resample_ema: float = 0.5                 # EMA smoothing over successive p
    resample_floor: float = 0.02              # epsilon floor per cluster
    resample_synth_per_cluster: int = 256     # N synthetic samples drawn per cluster
    # Delta-1 gate:
    use_physics_gate: bool = True             # Delta 1 enabled
    gate_cov_ema: float = 0.99                # running-covariance EMA for Mahalanobis dist
    # logging / housekeeping:
    log_every: int = 10
    eval_every: int = 25
    grad_clip: float = 0.0


@dataclass
class EvalConfig:
    n_scenarios: int = 64                     # ensemble size for ES / CRPS estimators
    mmd_bandwidths: tuple = (0.5, 1.0, 2.0, 5.0)  # multi-bandwidth RBF MMD
    tail_quantile: float = 0.95               # top-5% ramp events -> Tail-ES
    feasibility_tol: float = 1e-4             # tolerance for the feasibility-rate check
    forecaster_epochs: int = 60               # downstream forecaster training epochs (A5)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 0
    device: str = "cpu"
    out_dir: str = "./runs/default"

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        def _coerce(klass, sub):
            return klass(**{k: v for k, v in (sub or {}).items()})

        return cls(
            data=_coerce(DataConfig, d.get("data")),
            model=_coerce(ModelConfig, d.get("model")),
            physics=_coerce(PhysicsConfig, d.get("physics")),
            structural=_coerce(StructuralConfig, d.get("structural")),
            train=_coerce(TrainConfig, d.get("train")),
            eval=_coerce(EvalConfig, d.get("eval")),
            seed=d.get("seed", 0),
            device=d.get("device", "cpu"),
            out_dir=d.get("out_dir", "./runs/default"),
        )

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))


def smoke_config() -> Config:
    """A tiny config used by the smoke test / CI so the whole pipeline
    runs end-to-end in seconds without touching the optimal-size budgets."""
    c = Config()
    c.data.n_samples = 256
    c.data.horizon = 16
    c.data.n_channels = 3
    c.data.n_clusters = 4
    c.model.attn_dim = 32
    c.model.attn_depth = 2
    c.model.d_z = 16
    c.train.epochs = 3
    c.train.batch_size = 32
    c.train.K = 1
    c.train.resample_synth_per_cluster = 32
    c.eval.n_scenarios = 16
    c.eval.forecaster_epochs = 5
    return c
