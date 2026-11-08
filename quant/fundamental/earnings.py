"""Earnings-surprise tracking: surprise, surprise %, standardised unexpected earnings (SUE) and the price
reaction around announcements."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def surprises(earnings: pd.DataFrame, sue_window: int = 8) -> pd.DataFrame:
    """Adds surprise, surprise_pct, beat and SUE (surprise / std of the previous `sue_window` surprises)."""
    df = earnings.copy().sort_values("report_date").reset_index(drop=True)
    df["surprise"] = df["eps_actual"] - df["eps_estimate"]
    denom = df["eps_estimate"].abs().replace(0.0, np.nan)
    df["surprise_pct"] = df["surprise"] / denom
    df["beat"] = (df["surprise"] > 0).astype("boolean")
    df.loc[df["eps_actual"].isna() | df["eps_estimate"].isna(), "beat"] = pd.NA
    prior_std = df["surprise"].shift(1).rolling(sue_window, min_periods=max(3, sue_window // 2)).std(ddof=1)
    df["sue"] = df["surprise"] / prior_std.replace(0.0, np.nan)
    return df


def earnings_summary(df: pd.DataFrame) -> dict[str, object]:
    reported = df.dropna(subset=["eps_actual", "eps_estimate"])
    if reported.empty:
        return {"reports": 0}
    s = surprises(reported) if "surprise" not in reported.columns else reported
    return {
        "reports": int(len(s)),
        "beat_rate": float(s["beat"].dropna().astype("float64").mean()),
        "avg_surprise_pct": float(s["surprise_pct"].mean()),
        "last_report": s["report_date"].iloc[-1],
        "last_surprise_pct": float(s["surprise_pct"].iloc[-1]),
        "last_sue": float(s["sue"].iloc[-1]) if pd.notna(s["sue"].iloc[-1]) else None,
    }


def post_earnings_returns(
    prices: pd.Series, report_dates: Sequence, windows: Sequence[int] = (1, 5, 20)
) -> pd.DataFrame:
    """Return from the last close before each report to k sessions after (k in windows)."""
    px = prices.dropna().sort_index()
    idx = pd.DatetimeIndex(px.index)
    rows = []
    for d in report_dates:
        d = pd.Timestamp(d)
        pos = idx.searchsorted(d, side="left")  # first session >= report date
        if pos == 0 or pos >= len(idx):
            continue
        base = px.iloc[pos - 1]
        row = {"report_date": d.date()}
        for k in windows:
            j = pos - 1 + k
            row[f"ret_{k}d"] = float(px.iloc[j] / base - 1.0) if j < len(px) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
