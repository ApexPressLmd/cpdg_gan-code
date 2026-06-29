# CPDG-GAN

**Controllable, Physically-Consistent, Downstream-Guided GAN** for renewable
scenario synthesis and downstream load-forecasting augmentation.

This repository is a faithful, *un-simplified* PyTorch reproduction of the
method described in the source paper outline. Every atom of the architecture
and both core innovations are implemented as specified — including the exact
equations and the two appendix algorithms — rather than approximated. The
paper's reported numbers are explicitly illustrative placeholders; this code
reproduces the **method and experimental protocol**, not those specific
figures.

---

## 1. What is implemented

The generator backbone is a WGAN-GP fusing five atoms and two innovations:

| Tag | Component | Where |
|-----|-----------|-------|
| A1 | Attention-intensive generator — axial self-attention over **both** the temporal and the site axis | `src/models/attention.py`, `src/models/generator.py`, `src/models/discriminator.py` |
| A6 | Mutual-information controllable latent axes `c_int` (InfoGAN-style recognition net, spectral-normalised) | `src/models/mi_estimator.py` |
| A9 | Meteorological-causal conditioning → external categorical condition `c_ext` via Granger-style causal screening + clustering | `src/data/meteo_causal.py` |
| A2 | Differentiable physical-feasibility penalty `L_phys` (ramp-rate, capacity, non-negativity) | `src/models/physics.py` |
| A5 | Downstream probabilistic forecaster producing a CRPS error signal (frozen in the inner loop, retrained in the outer loop) | `src/models/forecaster.py` |
| Δ1 | **Diagnostic-Driven Physics-Gate** — Eq. (3), (5) | `src/models/physics_gate.py` |
| Δ2 | **Forecast-error-guided condition resampling** — Eq. (7) | `src/training/resampling.py` |

### The equations

(Equation numbers follow the manuscript: Eq. (1) is the WGAN-GP critic loss,
Eq. (2) the physical residual `r_phys`, Eq. (4) the Mahalanobis distance, and
Eq. (8) the full generator objective.)

* **Eq. (3)** gate: `g = σ(W · [ r_phys.detach(), dist(c_int, μ) ] + b)`.
  `r_phys` is **detached** for the gate; `dist` is a Mahalanobis distance using
  a running-covariance EMA of `c_int`; `W` is a single `1×2` affine map and `b`
  a scalar (no sub-network). See `PhysicsGate.forward`.
* **Eq. (5)** gated regulariser: `L_reg = (1 − g) · L_phys + g · L_struct`,
  evaluated **per sample** so the per-sample gate weights per-sample losses
  exactly. See `PhysicsGate.gated_regularizer`, `physics.residual_per_sample`,
  `structural.per_sample`.
* **Eq. (7)** condition resampling: the per-cluster error `e(c_ext)` (Eq. (6))
  is mapped to `softmax( e(c_ext) / τ )`, recomputed every `K` epochs,
  EMA-smoothed and ε-floored, then renormalised; `τ → ∞` ⇒ uniform.
  See `ConditionSampler.update`.

Generator objective:
`L_G = L_adv + λ_reg · L_reg + λ_mi · L_mi + λ_orth · L_orth`,
where `L_orth` decorrelates the **disjoint, concatenated** `c_ext` / `c_int`
embedding blocks (`src/training/losses.py`), and the structural invariants
`L_struct` combine patch-mask reconstruction, autocorrelation matching and
trend continuity (`src/models/structural.py`).

### The algorithms

* **Algorithm 1** (inner training step): `n_critic` critic updates followed by
  one generator update with the gated regulariser, MI and orthogonality terms —
  `Trainer.train_step`.
* **Algorithm 2** (outer resampling step): synthesise per cluster, retrain the
  forecaster on real + synthetic, measure per-cluster validation CRPS, update
  the condition sampler — `Trainer.outer_resample_step`.

Both are in `src/training/trainer.py`.

---

## 2. Repository layout

```
cpdg_gan/
├── src/
│   ├── utils/        seeding, dataclass-based config (Table A2 defaults)
│   ├── data/         causal conditioning (A9), preprocessing, datasets, pipeline
│   ├── models/       attention, generator, discriminator, MI (A6), physics (A2),
│   │                 structural, physics-gate (Δ1), forecaster (A5)
│   ├── training/     WGAN-GP losses, orthogonality, resampling (Δ2), Trainer
│   └── eval/         metrics (Energy Score, MMD, Tail-ES, feasibility, CRPS),
│                     evaluation harness
├── baselines/        WGAN-GP, ScenGAN(P1), PI-ST-GAN(P2), VAE-GAN(P5),
│                     TimeGAN, TS-Diffusion + registry
├── configs/          default.yaml + per-dataset variants + smoke.yaml + grid
├── scripts/          train / evaluate / run_ablations / run_baselines /
│                     make_figures / hpo
└── tests/            end-to-end smoke test
```

The "fusion-without-Δ" comparison row is **not** a separate baseline: it is the
full `Trainer` run with `AblationFlags.from_name("-delta1delta2")`.

---

## 3. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU-only torch:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Quick start

Everything runs out-of-the-box on synthetic data whose generative process
contains genuinely causal meteorological drivers plus nuisance covariates, so
the Granger screen (A9) has real signal to find. The training/splitting/
normalisation/conditioning/metric protocol is identical to the real-data path.

```bash
# Tiny end-to-end run (seconds, CPU) — trains, evaluates, exercises every path
python tests/test_smoke.py

# Full training with the paper's optimal hyper-parameters (Table A2)
PYTHONPATH=. python scripts/train.py --config configs/default.yaml --out runs/wind

# Evaluate a trained run (Energy Score, MMD, Tail-ES, feasibility, CRPS↓)
PYTHONPATH=. python scripts/evaluate.py --run runs/wind

# Table 5 ablations: full, -delta1, -delta2, -delta1delta2 (=fusion-no-Δ), -a6, -a9
PYTHONPATH=. python scripts/run_ablations.py --config configs/default.yaml

# Tables 3–4 baselines (roster + comparison)
PYTHONPATH=. python scripts/run_baselines.py --config configs/default.yaml

# Figures 5–9 from saved results
PYTHONPATH=. python scripts/make_figures.py --run runs/wind

# Hyper-parameter search over the Table A2 grid
PYTHONPATH=. python scripts/hpo.py --config configs/default.yaml
```

Use `--smoke` on `train.py`/`evaluate.py` for the fast CI config, and
`--ablation <name>` to train a specific variant.

### Using real data

`src/data/datasets.py::load_real` expects an `.npz` at `{root}/{name}.npz` with
arrays `power` of shape `(N, T, M)` and `meteo` of shape `(N, T, P)`. The three
named public benchmarks of Table 2 (DataCite-verified DOIs) are:

| Code | Config | Source (DOI) |
|------|--------|--------------|
| `wtk`  | `configs/wtk.yaml`  | NREL WIND Toolkit — https://doi.org/10.7799/1329290 |
| `swus` | `configs/swus.yaml` | NSRDB+WIND U.S.-regions collection (Mendeley) — https://doi.org/10.17632/x6r9c6zvw6 |
| `hdw`  | `configs/hdw.yaml`  | IEEE-DataPort hourly demand–weather — https://doi.org/10.21227/fpqq-nr70 |

Their archives are not downloadable in this environment, hence the synthetic
generators ship alongside the loader (identical split/normalisation/conditioning/
metric protocol). Save a prepared `.npz` to `{root}/{name}.npz` and point the
relevant config at it to run on the real datasets.

---

## 5. Hyper-parameters (defaults = Table A2 optimal)

`λ_reg=1.0, λ_mi=0.1, λ_orth=0.1, τ=0.5, K=10`, attention heads `=4`,
attention dim `=128`; WGAN-GP `n_critic=5, gp_lambda=10`, Adam `betas=(0.5,0.9)`,
`lr=2e-4`, `epochs=250`, `batch=64`. All live in `src/utils/config.py` and the
YAML files; the search grid is in `configs/search_space.py`.

---

## 6. Reproducibility (Section 4.1)

All randomness is seeded through `src/utils/seed.py` (seeds `{0..4}`); dataloader
workers are seeded via `worker_init_fn`. Multi-seed metrics (downstream CRPS
reduction) average over seeds. Configs are serialised next to every run
(`config.yaml`) so any result can be regenerated from its saved config and seed.
The code is device-agnostic and defaults to CPU.
