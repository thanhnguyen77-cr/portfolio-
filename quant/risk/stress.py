"""Historical stress windows and hypothetical shocks applied to a fixed-weight portfolio.

Windows are dated from public market history (peak-to-trough or event windows, inclusive). A window the
data does not cover is reported with its coverage, never dropped silently. Portfolio returns inside a
window hold the given weights constant (daily rebalanced), which is a simplification stated in the output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressWindow:
    name: str
    start: date
    end: date
    description: str
    region: str = "global"


STRESS_WINDOWS: dict[str, StressWindow] = {
    "gfc_2008": StressWindow(
        "gfc_2008", date(2008, 9, 1), date(2009, 3, 9), "Lehman to the March 2009 equity low", "global"
    ),
    "euro_crisis_2011": StressWindow(
        "euro_crisis_2011",
        date(2011, 7, 1),
        date(2011, 10, 4),
        "US downgrade and euro sovereign stress",
        "global",
    ),
    "taper_tantrum_2013": StressWindow(
        "taper_tantrum_2013",
        date(2013, 5, 22),
        date(2013, 6, 24),
        "Fed taper talk: rates up, EM and bonds down",
        "us",
    ),
    "brexit_2016": StressWindow(
        "brexit_2016",
        date(2016, 6, 23),
        date(2016, 7, 6),
        "UK referendum: GBP collapse, UK domestic stocks hit",
        "uk",
    ),
    "volmageddon_2018": StressWindow(
        "volmageddon_2018", date(2018, 1, 26), date(2018, 2, 8), "VIX spike and short-vol unwind", "us"
    ),
    "covid_2020": StressWindow(
        "covid_2020", date(2020, 2, 19), date(2020, 3, 23), "COVID crash, peak to trough", "global"
    ),
    "rates_2022": StressWindow(
        "rates_2022",
        date(2022, 1, 3),
        date(2022, 10, 12),
        "Inflation and rate shock: stocks and bonds down together",
        "global",
    ),
    "gilt_crisis_2022": StressWindow(
        "gilt_crisis_2022",
        date(2022, 9, 22),
        date(2022, 10, 14),
        "UK mini-budget: gilt yields spike, LDI unwind",
        "uk",
    ),
}


def window_returns(
    prices: pd.DataFrame, window: StressWindow, min_coverage: float = 0.8
) -> tuple[pd.DataFrame, float]:
    """Daily simple returns inside the window and the share of its business days with data."""
    px = pd.DataFrame(prices).sort_index()
    inside = px[(px.index >= pd.Timestamp(window.start)) & (px.index <= pd.Timestamp(window.end))]
    expected = len(pd.bdate_range(window.start, window.end))
    coverage = len(inside.dropna(how="all")) / expected if expected else 0.0
    rets = inside.pct_change().iloc[1:] if len(inside) > 1 else inside.iloc[0:0]
    return rets, float(coverage)


def historical_stress(
    weights: pd.Series | Mapping[str, float],
    prices: pd.DataFrame,
    windows: Mapping[str, StressWindow] | None = None,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """Portfolio and per-asset returns over each window with constant weights; low coverage is flagged."""
    w = pd.Series(weights, dtype=float)
    px = pd.DataFrame(prices).reindex(columns=w.index)
    rows = []
    for name, win in (windows or STRESS_WINDOWS).items():
        rets, cov = window_returns(px, win, min_coverage)
        row: dict[str, object] = {
            "window": name,
            "start": win.start,
            "end": win.end,
            "sessions": len(rets),
            "coverage": cov,
            "usable": cov >= min_coverage and len(rets) > 0,
            "description": win.description,
        }
        if row["usable"]:
            port = (rets.fillna(0.0) * w).sum(axis=1)
            row["portfolio_return"] = float(np.prod(1 + port) - 1)
            row["worst_day"] = float(port.min())
            path = np.r_[1.0, (1 + port).cumprod().to_numpy()]  # peak may be the window's starting value
            row["max_drawdown"] = float((path / np.maximum.accumulate(path) - 1).min())
            for a in w.index:
                row[f"ret_{a}"] = (
                    float(np.prod(1 + rets[a].dropna()) - 1) if rets[a].notna().any() else np.nan
                )
        else:
            row["portfolio_return"] = np.nan
            row["worst_day"] = np.nan
            row["max_drawdown"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def hypothetical_shocks(
    weights: pd.Series | Mapping[str, float], shocks: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    """Instant P&L of named shock sets {scenario: {asset: return}}; assets not shocked contribute zero."""
    w = pd.Series(weights, dtype=float)
    rows = []
    for name, shock in shocks.items():
        s = pd.Series(shock, dtype=float).reindex(w.index).fillna(0.0)
        contrib = w * s
        unknown = sorted(set(shock) - set(w.index))
        rows.append(
            {
                "scenario": name,
                "portfolio_return": float(contrib.sum()),
                **{f"pnl_{a}": float(v) for a, v in contrib.items()},
                "ignored_assets": ", ".join(unknown),
            }
        )
    return pd.DataFrame(rows)


def beta_shock(
    weights: pd.Series | Mapping[str, float], betas: pd.Series | Mapping[str, float], market_shock: float
) -> float:
    """Portfolio return for a market move via asset betas: sum(w_i * beta_i) * shock."""
    w = pd.Series(weights, dtype=float)
    b = pd.Series(betas, dtype=float).reindex(w.index)
    if b.isna().any():
        raise ValueError(f"betas missing for {list(b[b.isna()].index)}")
    return float((w * b).sum() * market_shock)
