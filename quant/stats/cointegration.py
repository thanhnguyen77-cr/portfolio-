"""Cointegration: Engle-Granger two-step, Johansen, spread half-life and z-score."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EngleGrangerResult:
    statistic: float
    pvalue: float
    critical_values: dict[str, float]
    hedge_ratio: float
    intercept: float
    spread: pd.Series
    half_life_sessions: float
    cointegrated_at_5pct: bool


@dataclass
class JohansenResult:
    trace_stats: list[float]
    trace_crit_95: list[float]
    max_eig_stats: list[float]
    max_eig_crit_95: list[float]
    rank_at_95: int
    eigenvectors: pd.DataFrame
    cointegrating_vector: pd.Series = field(default_factory=pd.Series)


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life from an AR(1) fit of the spread change on its lagged level (sessions)."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = s.diff().dropna()
    lag, delta = lag.align(delta, join="inner")
    x = np.column_stack([np.ones(len(lag)), lag.to_numpy()])
    beta = np.linalg.lstsq(x, delta.to_numpy(), rcond=None)[0][1]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2.0) / np.log(1.0 + beta))


def engle_granger(
    y: pd.Series, x: pd.Series, trend: str = "c", autolag: str | None = "aic"
) -> EngleGrangerResult:
    """Regress y on x, test the residual for a unit root (statsmodels.coint); hedge ratio from that OLS."""
    from statsmodels.tsa.stattools import coint

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 30:
        raise ValueError("need at least 30 aligned observations")
    stat, p, crit = coint(df["y"], df["x"], trend=trend, autolag=autolag)
    design = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    intercept, beta = np.linalg.lstsq(design, df["y"].to_numpy(), rcond=None)[0]
    spread = (df["y"] - beta * df["x"] - intercept).rename("spread")
    return EngleGrangerResult(
        float(stat),
        float(p),
        {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])},
        float(beta),
        float(intercept),
        spread,
        half_life(spread),
        bool(p < 0.05),
    )


def johansen(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen trace / max-eigenvalue tests; rank at 95% and the first cointegrating vector (normalised)."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    df = prices.dropna()
    if df.shape[1] < 2 or len(df) < 30:
        raise ValueError("need >= 2 series and >= 30 observations")
    res = coint_johansen(df.to_numpy(), det_order, k_ar_diff)
    trace, trace_crit = res.lr1, res.cvt[:, 1]
    maxeig, maxeig_crit = res.lr2, res.cvm[:, 1]
    rank = 0
    for stat, crit in zip(trace, trace_crit, strict=True):
        if stat > crit:
            rank += 1
        else:
            break
    vectors = pd.DataFrame(res.evec, index=df.columns, columns=[f"vec{i}" for i in range(res.evec.shape[1])])
    first = vectors["vec0"] / vectors["vec0"].iloc[0]
    return JohansenResult(
        [float(v) for v in trace],
        [float(v) for v in trace_crit],
        [float(v) for v in maxeig],
        [float(v) for v in maxeig_crit],
        rank,
        vectors,
        first.rename("cointegrating_vector"),
    )


def spread_zscore(spread: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score of the spread: rolling over `window` (as-of safe) or full-sample when None (descriptive)."""
    if window is None:
        return ((spread - spread.mean()) / spread.std(ddof=1)).rename("zscore")
    mu = spread.rolling(window, min_periods=window).mean()
    sd = spread.rolling(window, min_periods=window).std(ddof=1)
    return ((spread - mu) / sd).rename("zscore")
