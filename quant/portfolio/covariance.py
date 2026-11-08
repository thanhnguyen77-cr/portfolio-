"""Covariance estimators for daily return panels: sample, exponentially weighted and Ledoit-Wolf shrinkage.

Every estimator returns a `CovarianceEstimate` whose matrix is annualised by `periods_per_year`, symmetric
and positive semi-definite (eigenvalues that numerical noise pushes below zero are clipped). Estimation
error is surfaced through `n_obs` versus the number of assets and the `notes`; nothing here forecasts
volatility - a covariance matrix estimated from history describes that history.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CovarianceEstimate:
    cov: pd.DataFrame
    method: str
    n_obs: int
    periods_per_year: int
    shrinkage: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def assets(self) -> list[str]:
        return [str(a) for a in self.cov.index]

    def vol(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov.to_numpy(dtype=float))), index=self.cov.index, name="vol")

    def corr(self) -> pd.DataFrame:
        return cov_to_corr(self.cov)


def returns_from_prices(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Period returns from a wide price panel; rows where every asset is NaN are dropped."""
    px = pd.DataFrame(prices).sort_index().astype(float)
    if kind == "simple":
        rets = px / px.shift(1) - 1.0
    elif kind == "log":
        rets = np.log(px / px.shift(1))
    else:
        raise ValueError("kind must be 'simple' or 'log'")
    return rets.iloc[1:].dropna(how="all")


def _clean(returns: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(returns).dropna(how="any").astype(float)
    if df.shape[1] < 1:
        raise ValueError("need at least one asset")
    if df.shape[0] < 3:
        raise ValueError("need at least three complete observations")
    return df


def _notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if df.shape[0] < 2 * df.shape[1]:
        notes.append(
            f"{df.shape[0]} observations for {df.shape[1]} assets: the sample covariance is poorly "
            "determined; prefer shrinkage (ledoit_wolf) or fewer assets"
        )
    return notes


def sample_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> CovarianceEstimate:
    df = _clean(returns)
    raw = np.atleast_2d(np.cov(df.to_numpy(), rowvar=False, ddof=1)) * periods_per_year
    cov = pd.DataFrame(raw, index=df.columns, columns=df.columns)
    return CovarianceEstimate(nearest_psd(cov), "sample", len(df), periods_per_year, None, _notes(df))


def ewma_cov(
    returns: pd.DataFrame, halflife: float = 60.0, periods_per_year: int = 252, demean: bool = True
) -> CovarianceEstimate:
    """Exponentially weighted covariance (RiskMetrics style) with weights normalised to one.

    The 1/(1 - sum(w^2)) factor is the weighted analogue of the n/(n-1) correction, so an infinite
    half-life reproduces the sample covariance.
    """
    if halflife <= 0:
        raise ValueError("halflife must be positive")
    df = _clean(returns)
    x = df.to_numpy()
    n = len(x)
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    if demean:
        x = x - (w[:, None] * x).sum(axis=0)
    raw = (x * w[:, None]).T @ x / (1.0 - float((w**2).sum()))
    cov = pd.DataFrame(raw * periods_per_year, index=df.columns, columns=df.columns)
    est = CovarianceEstimate(nearest_psd(cov), "ewma", n, periods_per_year, None, _notes(df))
    est.notes.append(f"ewma half-life {halflife:g} periods; effective sample size {1.0 / (w**2).sum():.1f}")
    return est


def ledoit_wolf_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> CovarianceEstimate:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity, via scikit-learn (divides by n, not n-1)."""
    from sklearn.covariance import LedoitWolf

    df = _clean(returns)
    lw = LedoitWolf(assume_centered=False).fit(df.to_numpy())
    cov = pd.DataFrame(lw.covariance_ * periods_per_year, index=df.columns, columns=df.columns)
    return CovarianceEstimate(
        nearest_psd(cov), "ledoit_wolf", len(df), periods_per_year, float(lw.shrinkage_), _notes(df)
    )


METHODS = {"sample": sample_cov, "ewma": ewma_cov, "ledoit_wolf": ledoit_wolf_cov}


def covariance(returns: pd.DataFrame, method: str = "ledoit_wolf", **kwargs) -> CovarianceEstimate:  # noqa: ANN003
    if method not in METHODS:
        raise ValueError(f"unknown covariance method {method!r}; choose from {sorted(METHODS)}")
    return METHODS[method](returns, **kwargs)


def nearest_psd(cov: pd.DataFrame, eps: float = 0.0) -> pd.DataFrame:
    """Symmetrise and clip negative eigenvalues at `eps` (a cheap projection, fine for noise-level fixes)."""
    a = np.atleast_2d(cov.to_numpy(dtype=float))
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    if vals.min() >= eps:
        return pd.DataFrame(a, index=cov.index, columns=cov.columns)
    fixed = (vecs * np.clip(vals, eps, None)) @ vecs.T
    return pd.DataFrame((fixed + fixed.T) / 2.0, index=cov.index, columns=cov.columns)


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    a = cov.to_numpy(dtype=float)
    sd = np.sqrt(np.clip(np.diag(a), 0.0, None))
    sd[sd == 0] = np.nan
    corr = a / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.clip(corr, -1.0, 1.0), index=cov.index, columns=cov.columns)


def align_weights(weights: pd.Series | Mapping[str, float], cov: pd.DataFrame) -> np.ndarray:
    """Weights as an array in `cov` order; assets missing from `weights` get zero."""
    w = pd.Series(weights, dtype=float).reindex(cov.index).fillna(0.0)
    return w.to_numpy(dtype=float)


def portfolio_vol(weights: pd.Series | Mapping[str, float], cov: pd.DataFrame) -> float:
    w = align_weights(weights, cov)
    var = float(w @ cov.to_numpy(dtype=float) @ w)
    return float(np.sqrt(max(var, 0.0)))
