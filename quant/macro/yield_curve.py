"""Yield-curve analytics on tenor series (percent). Nelson-Siegel fitting is deferred; slope, curvature and
inversion flags cover the regime questions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def slope(long: pd.Series, short: pd.Series) -> pd.Series:
    """Long minus short tenor (percentage points), forward-filled onto the union of dates."""
    idx = long.index.union(short.index)
    return (long.reindex(idx).ffill() - short.reindex(idx).ffill()).dropna().rename("slope")


def curvature(short: pd.Series, mid: pd.Series, long: pd.Series) -> pd.Series:
    """Butterfly: 2 x mid - short - long (positive = humped)."""
    idx = short.index.union(mid.index).union(long.index)
    s, m, lg = (x.reindex(idx).ffill() for x in (short, mid, long))
    return (2 * m - s - lg).dropna().rename("curvature")


def inversion_flags(spread: pd.Series) -> pd.DataFrame:
    """Inverted flag (spread < 0), consecutive sessions inverted and un-inversion events.

    Un-inversion after a long inversion is a classic recession-watch flag; it is reported, not interpreted.
    """
    s = spread.dropna()
    inverted = s < 0
    run = inverted.astype(int).groupby((~inverted).cumsum()).cumsum()
    uninverted = (~inverted) & inverted.shift(1, fill_value=False)
    return pd.DataFrame(
        {"spread": s, "inverted": inverted, "sessions_inverted": run, "uninversion_event": uninverted}
    )


def curve_snapshot(tenors: dict[str, pd.Series], as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Latest level, 1-month and 3-month changes (bp) per tenor at or before `as_of`."""
    rows = []
    for name, series in tenors.items():
        s = series.dropna()
        if as_of is not None:
            s = s[s.index <= pd.Timestamp(as_of)]
        if s.empty:
            continue
        last = s.iloc[-1]
        m1 = s[s.index <= s.index[-1] - pd.Timedelta(days=30)]
        m3 = s[s.index <= s.index[-1] - pd.Timedelta(days=91)]
        rows.append(
            {
                "tenor": name,
                "date": s.index[-1].date(),
                "level": float(last),
                "chg_1m_bp": float((last - m1.iloc[-1]) * 100) if len(m1) else np.nan,
                "chg_3m_bp": float((last - m3.iloc[-1]) * 100) if len(m3) else np.nan,
            }
        )
    return pd.DataFrame(rows)
