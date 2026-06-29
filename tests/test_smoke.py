"""End-to-end smoke test for the whole CPDG-GAN pipeline.

Runs on the tiny :func:`smoke_config` so the full data -> train -> evaluate path
(every atom, both innovations, all ablations, and every baseline) executes in
seconds on CPU.  This is the fast CI gate that catches shape / wiring bugs.

Run:
    PYTHONPATH=. python tests/test_smoke.py
"""
from __future__ import annotations

import sys
import traceback

import torch

from src.utils.config import smoke_config
from src.utils.seed import set_seed
from src.data.datasets import prepare_data
from src.training.trainer import Trainer, AblationFlags
from src.eval.evaluate import evaluate_model, efficiency_summary
from baselines import REGISTRY, build_baseline


def _check_metrics(m):
    for k in ["ES", "MMD", "Tail_ES", "feasibility_rate",
              "CRPS_reduction_overall", "CRPS_reduction_hard"]:
        assert k in m, f"missing metric {k}"
        assert m[k] == m[k], f"NaN metric {k}"  # NaN check


def test_full_pipeline():
    cfg = smoke_config()
    set_seed(cfg.seed)
    bundle = prepare_data(cfg)
    assert bundle.T == cfg.data.horizon
    assert bundle.M == cfg.data.n_channels
    assert bundle.n_clusters >= 1

    trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name("full"))
    trainer.fit(verbose=False)

    # generation shape
    c = torch.zeros(8, dtype=torch.long)
    gen = trainer.generate(c)
    assert gen.shape == (8, bundle.T, bundle.M), gen.shape

    m = evaluate_model(trainer, bundle, cfg, verbose=False)
    _check_metrics(m)
    eff = efficiency_summary(trainer, cfg)
    assert eff["params_M"] > 0
    print(f"  [full] ES={m['ES']:.4f} feas={m['feasibility_rate']:.3f} "
          f"params={eff['params_M']:.3f}M")


def test_ablations():
    for name in ["-delta1", "-delta2", "-delta1delta2", "-a6", "-a9"]:
        cfg = smoke_config()
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        trainer = Trainer(cfg, bundle, flags=AblationFlags.from_name(name))
        trainer.fit(verbose=False)
        m = evaluate_model(trainer, bundle, cfg, verbose=False)
        _check_metrics(m)
        print(f"  [{name}] ES={m['ES']:.4f} feas={m['feasibility_rate']:.3f}")


def test_baselines():
    for key in REGISTRY:
        cfg = smoke_config()
        set_seed(cfg.seed)
        bundle = prepare_data(cfg)
        model = build_baseline(key, cfg, bundle, device=cfg.device)
        model.fit(verbose=False)
        c = torch.zeros(6, dtype=torch.long)
        gen = model.generate(c)
        assert gen.shape == (6, bundle.T, bundle.M), (key, gen.shape)
        m = evaluate_model(model, bundle, cfg, verbose=False)
        _check_metrics(m)
        print(f"  [{model.name}] ES={m['ES']:.4f} feas={m['feasibility_rate']:.3f}")


def main():
    tests = [test_full_pipeline, test_ablations, test_baselines]
    failures = 0
    for t in tests:
        print(f"\n=== {t.__name__} ===")
        try:
            t()
            print(f"  PASS: {t.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL: {t.__name__}")
            traceback.print_exc()
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
