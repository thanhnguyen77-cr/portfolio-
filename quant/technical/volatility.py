"""Volatility indicators. Wilder's ATR is seeded with the simple mean of the first `period` true ranges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.technical.trend import ema, sma


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0])
    return tr.rename("true_range")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder average true range; NaN before `period` observations."""
    if period < 1:
        raise ValueError("period must be >= 1")
    tr = true_range(high, low, close).to_numpy(dtype="float64")
    out = np.full(len(tr), np.nan)
    if len(tr) < period:
        return pd.Series(out, index=close.index, name=f"atr{period}")
    out[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(out, index=close.index, name=f"atr{period}")


def bollinger(prices: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(prices, window)
    sd = prices.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + n_std * sd, mid - n_std * sd
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_bandwidth": (upper - lower) / mid,
            "bb_pct_b": (prices - lower) / (upper - lower),
        }
    )


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_window: int = 20,
    atr_window: int = 10,
    mult: float = 2.0,
) -> pd.DataFrame:
    mid = ema(close, ema_window)
    band = mult * atr(high, low, close, atr_window)
    return pd.DataFrame({"kc_mid": mid, "kc_upper": mid + band, "kc_lower": mid - band})


def realised_vol(prices: pd.Series, window: int = 21, periods_per_year: int = 252) -> pd.Series:
    """Annualised standard deviation of log returns over a rolling window."""
    logret = np.log(prices).diff()
    return (logret.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)).rename(
        f"rv{window}"
    )


def parkinson_vol(
    high: pd.Series, low: pd.Series, window: int = 21, periods_per_year: int = 252
) -> pd.Series:
    """Parkinson range-based volatility, annualised."""
    hl = np.log(high / low) ** 2
    var = hl.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0))
    return (np.sqrt(var) * np.sqrt(periods_per_year)).rename(f"parkinson{window}")


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return (atr(high, low, close, period) / close).rename(f"atr{period}_pct")
