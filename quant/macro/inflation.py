"""Inflation transforms on monthly price indices and rate series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def yoy(index_level: pd.Series, periods: int = 12) -> pd.Series:
    """Year-over-year percentage change of a price index (percent)."""
    return (index_level.pct_change(periods) * 100.0).rename("yoy")


def mom(index_level: pd.Series) -> pd.Series:
    return (index_level.pct_change() * 100.0).rename("mom")


def annualised_3m(index_level: pd.Series) -> pd.Series:
    """Three-month change, annualised (percent): the near-term inflation impulse."""
    return (((index_level / index_level.shift(3)) ** 4 - 1.0) * 100.0).rename("ann_3m")


def real_rate(nominal: pd.Series, inflation_yoy: pd.Series) -> pd.Series:
    """Ex-post real rate: nominal yield minus the latest known year-over-year inflation (both percent)."""
    infl = inflation_yoy.dropna().reindex(nominal.index.union(inflation_yoy.dropna().index)).ffill()
    return (nominal - infl.reindex(nominal.index)).dropna().rename("real_rate")


def breakeven(nominal: pd.Series, real: pd.Series) -> pd.Series:
    """Market-implied inflation: nominal minus real (TIPS / index-linked) yield."""
    idx = nominal.index.union(real.index)
    return (nominal.reindex(idx).ffill() - real.reindex(idx).ffill()).dropna().rename("breakeven")


def inflation_regime(inflation: pd.Series, target: float = 2.0, lookback: int = 3) -> pd.DataFrame:
    """Level vs target and direction over `lookback` observations -> labels such as 'above_target_rising'."""
    s = inflation.dropna()
    change = s.diff(lookback)
    level = np.where(
        s > target + 0.5, "above_target", np.where(s < target - 0.5, "below_target", "at_target")
    )
    direction = np.where(change > 0.2, "rising", np.where(change < -0.2, "falling", "stable"))
    return pd.DataFrame(
        {
            "inflation": s,
            "change": change,
            "regime": [f"{lv}_{d}" for lv, d in zip(level, direction, strict=True)],
        }
    )
