"""Support and resistance from swing pivots.

A swing high is a bar whose high exceeds the highs of `left` bars before and `right` bars after it (so a
pivot is only known `right` bars later - callers must respect that lag). Pivot prices within `tolerance`
of each other are clustered into one level; the level's strength is its touch count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def swing_points(high: pd.Series, low: pd.Series, left: int = 5, right: int = 5) -> pd.DataFrame:
    """Columns swing_high / swing_low hold the pivot price on pivot bars (NaN elsewhere)."""
    n = len(high)
    h, lo = high.to_numpy(dtype="float64"), low.to_numpy(dtype="float64")
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(left, n - right):
        window_h = h[i - left : i + right + 1]
        window_l = lo[i - left : i + right + 1]
        if h[i] >= window_h.max() and np.argmax(window_h) == left:
            sh[i] = h[i]
        if lo[i] <= window_l.min() and np.argmin(window_l) == left:
            sl[i] = lo[i]
    return pd.DataFrame({"swing_high": sh, "swing_low": sl}, index=high.index)


def support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    left: int = 5,
    right: int = 5,
    tolerance: float = 0.01,
    min_touches: int = 2,
) -> pd.DataFrame:
    """Clustered pivot levels: level, touches, first/last touch, kind (support/resistance vs last close)."""
    pivots = swing_points(high, low, left, right)
    points = []
    for col in ("swing_high", "swing_low"):
        s = pivots[col].dropna()
        points += [(float(p), idx) for idx, p in s.items()]
    if not points:
        return pd.DataFrame(columns=["level", "touches", "first_touch", "last_touch", "kind", "distance_pct"])
    points.sort(key=lambda t: t[0])
    clusters: list[list[tuple[float, object]]] = [[points[0]]]
    for p in points[1:]:
        ref = np.mean([q[0] for q in clusters[-1]])
        if abs(p[0] - ref) / ref <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    last = float(close.iloc[-1])
    rows = []
    for cl in clusters:
        if len(cl) < min_touches:
            continue
        level = float(np.mean([q[0] for q in cl]))
        dates = [q[1] for q in cl]
        rows.append(
            {
                "level": level,
                "touches": len(cl),
                "first_touch": min(dates),
                "last_touch": max(dates),
                "kind": "resistance" if level > last else "support",
                "distance_pct": level / last - 1.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["touches", "last_touch"], ascending=[False, False]).reset_index(drop=True)


def nearest_levels(levels: pd.DataFrame, price: float, n: int = 2) -> dict[str, list[float]]:
    if levels.empty:
        return {"support": [], "resistance": []}
    below = levels[levels["level"] <= price].sort_values("level", ascending=False)["level"].head(n).tolist()
    above = levels[levels["level"] > price].sort_values("level")["level"].head(n).tolist()
    return {"support": below, "resistance": above}
