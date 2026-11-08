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


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    line = ema(prices, fast) - ema(prices, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Wilder's directional movement system: +DI, -DI and ADX in [0, 100]."""
    import numpy as np

    from quant.technical.volatility import true_range

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    tr = true_range(high, low, close)
    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / tr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / tr_s
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_line = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


def indicator_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """A standard indicator table for one instrument's bars (open/high/low/close/volume, sorted)."""
    from quant.technical.momentum import rsi
    from quant.technical.volatility import atr, bollinger, realised_vol
    from quant.technical.volume import relative_volume

    c, h, lo, v = bars["close"], bars["high"], bars["low"], bars["volume"]
    out = pd.DataFrame(index=bars.index)
    for w in (20, 50, 200):
        out[f"sma{w}"] = sma(c, w)
    out["ema20"] = ema(c, 20)
    out["rsi14"] = rsi(c, 14)
    out["rsi2"] = rsi(c, 2)
    out["atr14"] = atr(h, lo, c, 14)
    out["atr14_pct"] = out["atr14"] / c
    out = out.join(macd(c)).join(adx(h, lo, c)).join(bollinger(c))
    out["rv21"] = realised_vol(c, 21)
    out["relvol20"] = relative_volume(v, 20)
    out["high252"] = h.rolling(252, min_periods=252).max()
    out["dist_high252"] = c / out["high252"] - 1.0
    return out
