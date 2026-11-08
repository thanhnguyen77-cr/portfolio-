"""Volume indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume: cumulative signed volume."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum().rename("obv")


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int | None = 20
) -> pd.Series:
    """Volume-weighted average of the typical price; rolling over `window`, cumulative when None."""
    typical = (high + low + close) / 3.0
    pv = typical * volume
    if window is None:
        return (pv.cumsum() / volume.cumsum()).rename("vwap")
    num = pv.rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return (num / den).rename(f"vwap{window}")


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money-flow index in [0, 100]."""
    typical = (high + low + close) / 3.0
    flow = typical * volume
    up = flow.where(typical > typical.shift(1), 0.0)
    down = flow.where(typical < typical.shift(1), 0.0)
    pos = up.rolling(period, min_periods=period).sum()
    neg = down.rolling(period, min_periods=period).sum()
    ratio = pos / neg.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    out = out.where(neg != 0, 100.0).where(pos.notna())
    return out.rename(f"mfi{period}")


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume relative to the trailing average (excluding today)."""
    avg = volume.shift(1).rolling(window, min_periods=window).mean()
    return (volume / avg).rename(f"relvol{window}")
