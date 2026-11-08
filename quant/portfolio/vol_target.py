"""Volatility targeting: scale a weight vector to an ex-ante target, or lever a return stream by trailing
realised volatility (with the leverage decided one period before it applies, so it uses only known data)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant.portfolio.covariance import portfolio_vol


def ex_ante_vol(weights: pd.Series | Mapping[str, float], cov: pd.DataFrame) -> float:
    return portfolio_vol(weights, cov)


def scale_to_target_vol(
    weights: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    target_vol: float,
    max_leverage: float = 2.0,
) -> tuple[pd.Series, float]:
    """(scaled weights, leverage) with leverage = min(target / ex-ante vol, max_leverage)."""
    if target_vol <= 0 or max_leverage <= 0:
        raise ValueError("target_vol and max_leverage must be positive")
    w = pd.Series(weights, dtype=float).reindex(cov.index).fillna(0.0)
    vol = portfolio_vol(w, cov)
    leverage = min(target_vol / vol, max_leverage) if vol > 0 else 0.0
    return (w * leverage).rename("weight"), float(leverage)


def realised_vol_leverage(
    returns: pd.Series,
    target_vol: float,
    window: int = 21,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
    periods_per_year: int = 252,
    lag: int = 1,
) -> pd.Series:
    """Leverage per period from trailing realised vol, shifted by `lag`: period t uses vol known at t-lag."""
    if lag < 1:
        raise ValueError("lag must be at least 1: leverage must be set before the return it applies to")
    r = pd.Series(returns, dtype=float)
    vol = r.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    lev = (target_vol / vol).clip(lower=min_leverage, upper=max_leverage).shift(lag)
    return lev.rename("leverage")


def apply_leverage(
    returns: pd.Series, leverage: pd.Series, financing_rate: float = 0.0, periods_per_year: int = 252
) -> pd.Series:
    """Levered returns: L * r - (L - 1) * financing per period (NaN where leverage is undefined)."""
    r = pd.Series(returns, dtype=float)
    lev = pd.Series(leverage, dtype=float).reindex(r.index)
    out = lev * r - (lev - 1.0) * (financing_rate / periods_per_year)
    return out.rename("levered_return")
