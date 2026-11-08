"""Regime detection.

- volatility_regime: trailing percentile of realised volatility (as-of safe: only past data is ranked);
- trend_regime: moving-average structure, optionally confirmed by ADX;
- rule_based_regime: the legacy desk rules generalised to any benchmark (score -> risk_on / caution /
  risk_off with a position-size multiplier), using a volatility index when available and realised
  volatility otherwise;
- markov_regimes: a k-state Markov-switching model of returns (statsmodels). Smoothed probabilities use the
  whole sample and therefore look ahead; `filtered` probabilities are the as-of-safe series.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.technical.trend import adx, sma
from quant.technical.volatility import realised_vol


def volatility_regime(
    prices: pd.Series, window: int = 21, lookback: int = 252, low_pct: float = 0.33, high_pct: float = 0.67
) -> pd.DataFrame:
    rv = realised_vol(prices, window)
    pct = rv.rolling(lookback, min_periods=max(window * 3, 60)).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True
    )
    label = pd.Series(
        np.where(pct >= high_pct, "high", np.where(pct <= low_pct, "low", "normal")), index=prices.index
    )
    label = label.where(pct.notna(), None)
    return pd.DataFrame({"realised_vol": rv, "vol_percentile": pct, "vol_regime": label})


def trend_regime(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    adx_threshold: float = 20.0,
) -> pd.DataFrame:
    f, s = sma(close, fast), sma(close, slow)
    up = (close > f) & (f > s)
    down = (close < f) & (f < s)
    label = pd.Series(np.where(up, "uptrend", np.where(down, "downtrend", "range")), index=close.index)
    out = pd.DataFrame({"sma_fast": f, "sma_slow": s, "trend_regime": label.where(s.notna(), None)})
    if high is not None and low is not None:
        strength = adx(high, low, close)["adx"]
        out["adx"] = strength
        weak = strength < adx_threshold
        out["trend_regime"] = out["trend_regime"].where(~weak | strength.isna(), "range")
    return out


@dataclass(frozen=True)
class RegimeRule:
    risk_on: float
    caution: float


def rule_based_regime(
    bench_close: pd.Series,
    vol_index: pd.Series | None = None,
    *,
    vol_window: int = 21,
    vol_lookback: int = 252,
) -> pd.DataFrame:
    """Score = (close>SMA50) + (close>SMA200) + (SMA50>SMA200) + vol term - drawdown penalty.

    Vol term: with a volatility index, +1 below 20, 0 for 20-30, -2 above 30; without one, +1 when realised
    vol sits in the bottom third of its trailing distribution, -2 in the top tenth. Drawdown from the 252-
    session high beyond 10% costs 1. Labels: score >= 3 risk_on (size 1.0), >= 1 caution (0.5), else
    risk_off (0.0).
    """
    f, s = sma(bench_close, 50), sma(bench_close, 200)
    score = (bench_close > f).astype(float) + (bench_close > s).astype(float) + (f > s).astype(float)
    if vol_index is not None:
        vi = vol_index.reindex(bench_close.index).ffill()
        vol_term = pd.Series(np.where(vi < 20, 1.0, np.where(vi > 30, -2.0, 0.0)), index=bench_close.index)
        vol_term = vol_term.where(vi.notna(), 0.0)
    else:
        pct = volatility_regime(bench_close, vol_window, vol_lookback)["vol_percentile"]
        vol_term = pd.Series(
            np.where(pct <= 0.33, 1.0, np.where(pct >= 0.90, -2.0, 0.0)), index=bench_close.index
        )
        vol_term = vol_term.where(pct.notna(), 0.0)
    high_252 = bench_close.rolling(252, min_periods=50).max()
    dd_pen = ((bench_close / high_252 - 1.0) < -0.10).astype(float)
    total = score + vol_term - dd_pen
    label = pd.Series(
        np.where(total >= 3, "risk_on", np.where(total >= 1, "caution", "risk_off")), index=bench_close.index
    )
    label = label.where(s.notna(), None)
    size = label.map({"risk_on": 1.0, "caution": 0.5, "risk_off": 0.0})
    return pd.DataFrame(
        {"score": total.where(s.notna()), "regime": label, "size_multiplier": size, "vol_term": vol_term}
    )


@dataclass
class MarkovResult:
    k_regimes: int
    means: list[float]
    variances: list[float]
    transition: np.ndarray
    smoothed: pd.DataFrame  # P(regime | all data): looks ahead, for description only
    filtered: pd.DataFrame  # P(regime | data up to t): as-of safe
    regime: pd.Series  # argmax of filtered probabilities, 0 = lowest variance
    converged: bool
    loglik: float


def markov_regimes(returns: pd.Series, k_regimes: int = 2, switching_variance: bool = True) -> MarkovResult:
    """Fit a Markov-switching mean/variance model; regimes are re-ordered by variance (0 = calm)."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    r = returns.dropna().astype("float64")
    scale = 100.0  # percent units help the optimiser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MarkovRegression(
            r.to_numpy() * scale, k_regimes=k_regimes, trend="c", switching_variance=switching_variance
        )
        fit = model.fit(disp=False, maxiter=500)
    params = dict(zip(model.param_names, np.asarray(fit.params, dtype="float64"), strict=True))
    means = [float(params[f"const[{i}]"]) / scale for i in range(k_regimes)]
    if switching_variance:
        variances = [float(params[f"sigma2[{i}]"]) / scale**2 for i in range(k_regimes)]
    else:
        variances = [float(params["sigma2"]) / scale**2] * k_regimes
    order = list(np.argsort(variances))
    smoothed = pd.DataFrame(np.asarray(fit.smoothed_marginal_probabilities)[:, order], index=r.index)
    filtered = pd.DataFrame(np.asarray(fit.filtered_marginal_probabilities)[:, order], index=r.index)
    smoothed.columns = filtered.columns = [f"regime_{i}" for i in range(k_regimes)]
    raw = np.asarray(fit.regime_transition)
    raw = raw[:, :, 0] if raw.ndim == 3 else raw
    # statsmodels stores P(to | from) with columns summing to one; present rows as the 'from' state
    transition = raw.T[np.ix_(order, order)]
    regime = pd.Series(filtered.to_numpy().argmax(axis=1), index=r.index, name="regime")
    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
    return MarkovResult(
        k_regimes,
        [means[i] for i in order],
        [variances[i] for i in order],
        transition,
        smoothed,
        filtered,
        regime,
        converged,
        float(fit.llf),
    )
