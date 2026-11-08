"""Momentum indicators. Wilder's RSI is seeded with the simple mean of the first `period` changes, exactly as
in the classic definition (the legacy stdlib implementation and its reference values are the oracle)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.technical.trend import sma


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100]; NaN for the first `period` observations."""
    if period < 1:
        raise ValueError("period must be >= 1")
    px = prices.to_numpy(dtype="float64")
    out = np.full(len(px), np.nan)
    if len(px) <= period:
        return pd.Series(out, index=prices.index, name=f"rsi{period}")
    delta = np.diff(px)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return pd.Series(out, index=prices.index, name=f"rsi{period}")


def roc(prices: pd.Series, period: int = 21) -> pd.Series:
    """Rate of change: price / price[t-period] - 1."""
    return prices.pct_change(period).rename(f"roc{period}")


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    """%K = (close - lowest low) / (highest high - lowest low) x 100 over `k`; %D = SMA(%K, d)."""
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    rng = (hh - ll).replace(0.0, np.nan)
    pct_k = ((close - ll) / rng * 100.0).clip(0, 100)
    pct_d = sma(pct_k, d)
    return pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_d})


def momentum_12_1(prices: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Classic 12-1 momentum: return over `lookback` sessions excluding the most recent `skip`."""
    return (prices.shift(skip) / prices.shift(lookback) - 1.0).rename("mom_12_1")
