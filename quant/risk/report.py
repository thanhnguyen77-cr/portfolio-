"""Assemble a risk report (VaR table, drawdowns, stress windows, correlation structure) as sectioned HTML."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.reports.builder import Section, plotly_js_source, render_report
from quant.risk.correlation import correlation_summary
from quant.risk.drawdown import drawdown_episodes, underwater_summary
from quant.risk.stress import STRESS_WINDOWS, historical_stress
from quant.risk.var import kupiec_test, rolling_historical_var, var_table


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series | Mapping[str, float]) -> pd.Series:
    """Constant-weight (daily rebalanced) portfolio returns; a simplification used for risk estimates."""
    w = pd.Series(weights, dtype=float)
    df = pd.DataFrame(returns).reindex(columns=w.index).dropna(how="any")
    return (df * w).sum(axis=1).rename("portfolio")


def _line_chart(series: pd.Series, name: str, hover: str = "%{y:.2%}") -> dict[str, Any]:
    idx = pd.DatetimeIndex(series.index)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": name,
                "x": [d.strftime("%Y-%m-%d") for d in idx],
                "y": [None if not np.isfinite(v) else float(v) for v in series.to_numpy(dtype=float)],
                "hovertemplate": hover + "<extra>" + name + "</extra>",
                "meta": {"role": "series", "slot": 0},
            }
        ],
        "layout": {"yaxis": {"tickformat": ".0%"}},
    }


def build_risk_report(
    returns: pd.DataFrame,
    weights: pd.Series | Mapping[str, float],
    *,
    prices: pd.DataFrame | None = None,
    cov: pd.DataFrame | None = None,
    alpha: float = 0.99,
    horizon: int = 1,
    var_window: int = 250,
    title: str = "Risk report",
    subtitle: str = "",
    extra_sections: Sequence[Section] = (),
    extra_caveats: Sequence[str] = (),
    path: Path | str | None = None,
    inline_plotly: bool = False,
) -> str:
    w = pd.Series(weights, dtype=float)
    port = portfolio_returns(returns, w)
    equity = (1 + port).cumprod() * 100.0
    caveats = [
        "Risk measures are estimated from the sample shown; they describe that history and will be wrong "
        "about the next crisis in ways the sample cannot reveal.",
        "Portfolio returns hold the given weights constant (daily rebalancing, no costs), which overstates "
        "diversification relative to a drifting buy-and-hold book.",
        f"VaR/CVaR at {alpha:.0%} over {horizon} day(s); parametric and simulation methods scale with the "
        "square root of time and assume iid returns.",
    ] + list(extra_caveats)
    sections: list[Section] = []

    vt = var_table(port, alpha, horizon)
    sections.append(
        Section(
            "Value at Risk",
            kv={
                "observations": len(port),
                "window": f"{port.index[0].date()} to {port.index[-1].date()}",
                "annualised vol": f"{port.std(ddof=1) * np.sqrt(252):.2%}",
            },
            table=vt,
        )
    )
    if len(port) > var_window + 50:
        rv = rolling_historical_var(port, var_window, alpha)
        k = kupiec_test(port, rv, alpha)
        sections.append(
            Section(
                "VaR backtest (Kupiec)",
                text=(
                    f"Rolling {var_window}-day historical VaR, out of sample: {k['exceptions']} exceptions "
                    "in "
                    f"{k['n']} days versus {k['expected']:.1f} expected; p-value {k['p_value']:.3f}"
                    + (" (rejected at 5%)." if k["reject_at_5pct"] else " (not rejected at 5%).")
                ),
                chart=_line_chart(-rv.dropna(), "-VaR (loss threshold)"),
            )
        )
    else:
        caveats.append(
            f"Sample too short for a rolling {var_window}-day VaR backtest; VaR is in-sample only."
        )

    uw = underwater_summary(equity)
    sections.append(
        Section(
            "Drawdowns",
            kv={k: (f"{v:.2%}" if isinstance(v, float) and abs(v) < 5 else v) for k, v in uw.items()},
            table=drawdown_episodes(equity, top=10),
            chart=_line_chart(equity / equity.cummax() - 1, "drawdown"),
        )
    )

    if prices is not None:
        st = historical_stress(w, prices, STRESS_WINDOWS)
        n_usable = int(st["usable"].sum())
        sections.append(
            Section(
                "Historical stress windows",
                text=f"{n_usable} of {len(st)} windows covered by the price history (others show NaN).",
                table=st.drop(columns=["description"]).assign(description=st["description"]),
            )
        )
        if n_usable < len(st):
            caveats.append(
                f"Only {n_usable}/{len(st)} stress windows fall inside the available price history."
            )

    cs = correlation_summary(returns, cov, w)
    sections.append(
        Section(
            "Correlation structure",
            kv={k: f"{v:.3f}" for k, v in cs.items()},
            table=pd.DataFrame(returns).dropna(how="any").corr().round(3).rename_axis("asset").reset_index(),
        )
    )
    sections += list(extra_sections)
    inline_js = plotly_js_source() if inline_plotly else None
    return render_report(title, sections, caveats, subtitle=subtitle, path=path, inline_js=inline_js)
