"""Drawdown analytics: episode table (peak, trough, recovery), underwater statistics, Calmar, ulcer and
pain indices, and conditional drawdown at risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.metrics import drawdown_series, max_drawdown

__all__ = [
    "calmar_ratio",
    "conditional_drawdown_at_risk",
    "drawdown_episodes",
    "drawdown_series",
    "max_drawdown",
    "pain_index",
    "ulcer_index",
    "underwater_summary",
]


def drawdown_episodes(equity: pd.Series, top: int | None = 10) -> pd.DataFrame:
    """One row per drawdown episode: peak, trough, recovery (NaT if still underwater), depth and lengths."""
    eq = pd.Series(equity, dtype=float).dropna()
    dd = drawdown_series(eq)
    rows = []
    in_dd = False
    peak_i = 0
    for i in range(len(dd)):
        if not in_dd and dd.iloc[i] < 0:
            in_dd = True
            peak_i = i - 1 if i > 0 else 0
            trough_i = i
        elif in_dd:
            if dd.iloc[i] < dd.iloc[trough_i]:
                trough_i = i
            if dd.iloc[i] >= 0:
                rows.append(_episode(eq, dd, peak_i, trough_i, i))
                in_dd = False
    if in_dd:
        rows.append(_episode(eq, dd, peak_i, trough_i, None))
    cols = [
        "peak",
        "trough",
        "recovery",
        "depth",
        "to_trough_sessions",
        "recovery_sessions",
        "total_sessions",
    ]
    table = pd.DataFrame(rows, columns=cols)
    table = table.sort_values("depth").reset_index(drop=True)
    return table.head(top) if top else table


def _episode(eq: pd.Series, dd: pd.Series, peak_i: int, trough_i: int, rec_i: int | None) -> dict:
    end_i = rec_i if rec_i is not None else len(eq) - 1
    return {
        "peak": eq.index[peak_i],
        "trough": eq.index[trough_i],
        "recovery": eq.index[rec_i] if rec_i is not None else pd.NaT,
        "depth": float(dd.iloc[trough_i]),
        "to_trough_sessions": int(trough_i - peak_i),
        "recovery_sessions": int(rec_i - trough_i) if rec_i is not None else np.nan,
        "total_sessions": int(end_i - peak_i),
    }


def underwater_summary(equity: pd.Series) -> dict[str, float | int]:
    eq = pd.Series(equity, dtype=float).dropna()
    dd = drawdown_series(eq)
    mdd, longest = max_drawdown(eq)
    return {
        "max_drawdown": mdd,
        "longest_underwater_sessions": longest,
        "time_underwater": float((dd < 0).mean()) if len(dd) else 0.0,
        "average_drawdown": float(dd[dd < 0].mean()) if (dd < 0).any() else 0.0,
        "current_drawdown": float(dd.iloc[-1]) if len(dd) else 0.0,
        "ulcer_index": ulcer_index(eq),
        "pain_index": pain_index(eq),
    }


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown_series(pd.Series(equity, dtype=float).dropna())
    return float(np.sqrt((dd**2).mean())) if len(dd) else 0.0


def pain_index(equity: pd.Series) -> float:
    dd = drawdown_series(pd.Series(equity, dtype=float).dropna())
    return float(-dd.mean()) if len(dd) else 0.0


def calmar_ratio(equity: pd.Series, periods_per_year: int = 252) -> float:
    eq = pd.Series(equity, dtype=float).dropna()
    if len(eq) < 2:
        return float("nan")
    years = (len(eq) - 1) / periods_per_year
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 and eq.iloc[0] > 0 else float("nan")
    mdd, _ = max_drawdown(eq)
    return float(cagr / abs(mdd)) if mdd < 0 else float("nan")


def conditional_drawdown_at_risk(equity: pd.Series, alpha: float = 0.95) -> float:
    """CDaR: mean of the worst (1 - alpha) share of daily drawdown observations (positive number)."""
    dd = -drawdown_series(pd.Series(equity, dtype=float).dropna())
    if dd.empty:
        return 0.0
    cut = float(dd.quantile(alpha))
    tail = dd[dd >= cut]
    return float(tail.mean()) if len(tail) else cut
