"""Cross-asset regime board.

Takes whatever series are available (growth, inflation, rates, credit, FX, equities, volatility), reports
each one's latest value, 3-month change and 5-year z-score, and places the economy in a growth x inflation
quadrant from the *direction* of growth and inflation impulses. Missing inputs are listed, never imputed.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

QUADRANTS = {
    ("up", "up"): "reflation / overheating",
    ("up", "down"): "goldilocks / disinflationary growth",
    ("down", "up"): "stagflation",
    ("down", "down"): "deflationary slowdown",
}

# indicator -> (category, higher_is_risk_on)
INDICATOR_META: dict[str, tuple[str, bool | None]] = {
    "growth_ip_yoy": ("growth", True),
    "growth_unemployment": ("growth", False),
    "growth_payrolls_3m": ("growth", True),
    "inflation_yoy": ("inflation", None),
    "inflation_3m_ann": ("inflation", None),
    "rates_policy": ("rates", None),
    "rates_10y": ("rates", None),
    "rates_slope_10y_2y": ("rates", True),
    "rates_real_10y": ("rates", None),
    "credit_hy_oas": ("credit", False),
    "credit_ig_oas": ("credit", False),
    "fx_dollar": ("fx", False),
    "fx_gbpusd": ("fx", None),
    "equity_vs_200d": ("equity", True),
    "vol_index": ("volatility", False),
}


def _latest_change_z(
    s: pd.Series, lookback_days: int = 91, z_years: int = 5
) -> tuple[float, float, float, pd.Timestamp]:
    s = s.dropna()
    last = float(s.iloc[-1])
    past = s[s.index <= s.index[-1] - pd.Timedelta(days=lookback_days)]
    change = last - float(past.iloc[-1]) if len(past) else np.nan
    hist = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * z_years)]
    z = (last - hist.mean()) / hist.std(ddof=1) if len(hist) > 24 and hist.std(ddof=1) > 0 else np.nan
    return last, change, float(z) if np.isfinite(z) else np.nan, s.index[-1]


def _direction(change: float, tol: float) -> str:
    if not np.isfinite(change):
        return "unknown"
    return "up" if change > tol else ("down" if change < -tol else "flat")


def growth_inflation_quadrant(growth_change: float, inflation_change: float, *, tol: float = 0.1) -> str:
    g, i = _direction(growth_change, tol), _direction(inflation_change, tol)
    if "unknown" in (g, i):
        return "unknown (missing inputs)"
    g = "up" if g in {"up", "flat"} else "down"
    i = "up" if i in {"up", "flat"} else "down"
    return QUADRANTS[(g, i)]


def cross_asset_dashboard(
    series: Mapping[str, pd.Series], *, as_of: pd.Timestamp | None = None
) -> dict[str, object]:
    """Latest level, 3-month change and 5-year z-score per indicator, plus the growth x inflation quadrant.

    `series` keys come from INDICATOR_META (missing ones are reported, never imputed). Returns a dict with
    'table', 'quadrant', 'missing' and 'risk_signals'.
    """
    rows = []
    missing = [k for k in INDICATOR_META if k not in series or series[k].dropna().empty]
    for key, s in series.items():
        if key not in INDICATOR_META or s.dropna().empty:
            continue
        s = s.dropna()
        if as_of is not None:
            s = s[s.index <= pd.Timestamp(as_of)]
        if s.empty:
            missing.append(key)
            continue
        last, change, z, when = _latest_change_z(s)
        category, risk_on_if_higher = INDICATOR_META[key]
        if risk_on_if_higher is None or not np.isfinite(z):
            signal = "neutral"
        else:
            tilt = z if risk_on_if_higher else -z
            signal = "risk_on" if tilt > 0.5 else ("risk_off" if tilt < -0.5 else "neutral")
        rows.append(
            {
                "indicator": key,
                "category": category,
                "as_of": when.date(),
                "value": last,
                "chg_3m": change,
                "z_5y": z,
                "signal": signal,
            }
        )
    table = pd.DataFrame(
        rows, columns=["indicator", "category", "as_of", "value", "chg_3m", "z_5y", "signal"]
    )
    growth_change = table.loc[table["indicator"] == "growth_ip_yoy", "chg_3m"]
    if growth_change.empty and "growth_unemployment" in table["indicator"].values:
        growth_change = -table.loc[table["indicator"] == "growth_unemployment", "chg_3m"]
    infl_change = (
        table.loc[table["indicator"] == "inflation_yoy", "chg_3m"]
        if not table.empty
        else pd.Series(dtype=float)
    )
    quadrant = growth_inflation_quadrant(
        float(growth_change.iloc[0]) if len(growth_change) else np.nan,
        float(infl_change.iloc[0]) if len(infl_change) else np.nan,
    )
    signals = table["signal"].value_counts().to_dict() if not table.empty else {}
    return {"table": table, "quadrant": quadrant, "missing": sorted(set(missing)), "risk_signals": signals}
