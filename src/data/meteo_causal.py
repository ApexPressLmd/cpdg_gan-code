"""A9 - Meteorological-causal conditioning.

Implements the front-end described in Section 3.5:

    "The external condition c_ext is built by extracting meteorological
     drivers, applying a causal-screening test to retain only factors with
     a directed influence on output, and clustering the survivors into
     discrete labels."

The causal-screening test is a Granger-style nested-model F-test: a covariate
w_j is retained iff adding its lagged values significantly reduces the
one-step-ahead prediction error of the target (aggregate power output) relative
to an autoregressive baseline.  The retained drivers' per-window summary
statistics are then clustered with k-means into the discrete labels c_ext.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _lag_matrix(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Return [x_{t-1}, ..., x_{t-max_lag}] aligned to t in [max_lag, N)."""
    n = series.shape[0]
    cols = [series[max_lag - l: n - l] for l in range(1, max_lag + 1)]
    return np.stack(cols, axis=1)


def granger_screen(
    target: np.ndarray,
    covariate: np.ndarray,
    max_lag: int = 3,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """Granger F-test: does ``covariate`` Granger-cause ``target``?

    Restricted model:  y_t ~ y_{t-1..p}
    Full model:        y_t ~ y_{t-1..p} + w_{t-1..p}
    Retained iff the F-statistic on the extra block is significant at ``alpha``.

    Returns (retained, p_value).
    """
    target = np.asarray(target, dtype=np.float64).ravel()
    covariate = np.asarray(covariate, dtype=np.float64).ravel()
    n = target.shape[0]
    y = target[max_lag:]
    yl = _lag_matrix(target, max_lag)
    wl = _lag_matrix(covariate, max_lag)

    # ordinary least squares with intercept
    def _ols_rss(X: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
        Xc = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        return float(resid @ resid), Xc.shape[1]

    rss_r, k_r = _ols_rss(yl, y)
    rss_f, k_f = _ols_rss(np.concatenate([yl, wl], axis=1), y)

    df1 = k_f - k_r                      # number of extra parameters (== max_lag)
    df2 = len(y) - k_f                   # residual dof of full model
    if df2 <= 0 or rss_f <= 0 or df1 <= 0:
        return False, 1.0
    f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return bool(p_value < alpha), p_value


@dataclass
class MeteoCausalConditioner:
    """Fit-once front-end that maps raw meteo covariates -> discrete c_ext.

    Parameters
    ----------
    n_clusters : number of discrete external conditions.
    max_lag    : lag order for the Granger screen.
    alpha      : significance level for retention.
    """

    n_clusters: int = 6
    max_lag: int = 3
    alpha: float = 0.05

    retained_: Optional[List[int]] = None
    p_values_: Optional[np.ndarray] = None
    _scaler: Optional[StandardScaler] = None
    _kmeans: Optional[KMeans] = None

    # ------------------------------------------------------------------ fit
    def fit(self, meteo: np.ndarray, power: np.ndarray) -> "MeteoCausalConditioner":
        """Fit the screen + clusterer.

        Parameters
        ----------
        meteo : (N, T, P) windowed meteorological covariates.
        power : (N, T, M) windowed power output (target).
        """
        N, T, P = meteo.shape
        # aggregate power over channels -> a single output series per window,
        # then concatenate windows into one long series for the causal screen.
        agg_power = power.mean(axis=2)               # (N, T)
        flat_power = agg_power.reshape(-1)           # (N*T,)
        flat_meteo = meteo.reshape(N * T, P)         # (N*T, P)

        retained, pvals = [], np.ones(P)
        for j in range(P):
            keep, pv = granger_screen(
                flat_power, flat_meteo[:, j], self.max_lag, self.alpha
            )
            pvals[j] = pv
            if keep:
                retained.append(j)
        # never return an empty set: fall back to the most significant driver.
        if not retained:
            retained = [int(np.argmin(pvals))]
        self.retained_ = retained
        self.p_values_ = pvals

        # window-level features from retained drivers: mean + std per driver.
        feats = self._window_features(meteo)
        self._scaler = StandardScaler().fit(feats)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters, n_init=10, random_state=0
        ).fit(self._scaler.transform(feats))
        return self

    def _window_features(self, meteo: np.ndarray) -> np.ndarray:
        sub = meteo[:, :, self.retained_]            # (N, T, |retained|)
        mean = sub.mean(axis=1)
        std = sub.std(axis=1)
        return np.concatenate([mean, std], axis=1)   # (N, 2*|retained|)

    # ------------------------------------------------------------- transform
    def transform(self, meteo: np.ndarray) -> np.ndarray:
        """Return integer cluster labels c_ext of shape (N,)."""
        assert self._kmeans is not None, "call fit() first"
        feats = self._scaler.transform(self._window_features(meteo))
        return self._kmeans.predict(feats).astype(np.int64)

    def fit_transform(self, meteo: np.ndarray, power: np.ndarray) -> np.ndarray:
        return self.fit(meteo, power).transform(meteo)
