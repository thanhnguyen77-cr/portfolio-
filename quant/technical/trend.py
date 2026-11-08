"""Trend indicators: price Series in, same-length Series out (NaN during warm-up)."""

from __future__ import annotations

import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return prices.rolling(window, min_periods=window).mean().rename(f"sma{window}")


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential moving average seeded with the first `span` observations' SMA (Wilder-style start)."""
    if span < 1:
        raise ValueError("span must be >= 1")
    out = prices.ewm(span=span, adjust=False, min_periods=span).mean()
    return out.rename(f"ema{span}")
