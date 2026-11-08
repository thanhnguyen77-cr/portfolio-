"""Backtest tearsheet: stat tiles, equity / drawdown / yearly-return / exposure charts, costs, robustness,
configuration and the mandatory caveats. Charts carry role metadata (series slot, polarity) instead of
colours; the page assigns validated palette colours per colour scheme at render time and offers a table view.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.backtest.metrics import drawdown_series
from quant.backtest.result import BacktestResult
from quant.reports.builder import Section, plotly_js_source, render_report


def _pct(v: object) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):+.2%}"


def _num(v: object, fmt: str = "{:.2f}") -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(float(v))


def _line(
    x: pd.Index, y: pd.Series, name: str, slot: int, *, fill: bool = False, hover: str = "%{y:,.2f}"
) -> dict:
    trace: dict[str, Any] = {
        "type": "scatter",
        "mode": "lines",
        "name": name,
        "x": [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(x)],
        "y": [None if not np.isfinite(v) else float(v) for v in y.to_numpy(dtype="float64")],
        "hovertemplate": hover + "<extra>" + name + "</extra>",
        "meta": {"role": "series", "slot": slot},
    }
    if fill:
        trace["fill"] = "tozeroy"
    return trace


def equity_section(result: BacktestResult) -> Section:
    eq = result.equity_series
    traces = [_line(eq.index, eq, "strategy", 0)]
    table = pd.DataFrame({"session_date": eq.index.date, "equity": eq.to_numpy()})
    if result.benchmark is not None and len(result.benchmark.dropna()) > 1:
        bench = result.benchmark.astype("float64").dropna()
        bench = bench[(bench.index >= eq.index[0]) & (bench.index <= eq.index[-1])]
        if len(bench) > 1:
            rebased = bench / bench.iloc[0] * eq.iloc[0]
            traces.append(_line(rebased.index, rebased, "benchmark (rebased)", 1))
            table = table.merge(
                pd.DataFrame({"session_date": rebased.index.date, "benchmark": rebased.to_numpy()}),
                how="left",
            )
    layout = {
        "yaxis": {"title": {"text": f"equity ({result.config.get('base_currency')})"}, "tickformat": ",.0f"}
    }
    return Section("Equity curve", chart={"data": traces, "layout": layout}, table=table, chart_height=340)


def drawdown_section(result: BacktestResult) -> Section:
    eq = result.equity_series
    dd = drawdown_series(eq)
    trace = _line(dd.index, dd, "drawdown", 0, fill=True, hover="%{y:.2%}")
    table = pd.DataFrame({"session_date": dd.index.date, "drawdown": dd.to_numpy()})
    layout = {"yaxis": {"title": {"text": "drawdown"}, "tickformat": ".0%"}}
    return Section("Drawdown", chart={"data": [trace], "layout": layout}, table=table, chart_height=240)


def yearly_section(result: BacktestResult) -> Section:
    by_year = result.metrics.get("returns_by_year") or {}
    years = [str(y) for y in by_year]
    values = [float(v) for v in by_year.values()]
    trace = {
        "type": "bar",
        "name": "return",
        "x": years,
        "y": values,
        "text": [f"{v:+.1%}" for v in values],
        "textposition": "outside",
        "hovertemplate": "%{x}: %{y:+.2%}<extra></extra>",
        "meta": {"role": "polarity"},
    }
    layout = {
        "xaxis": {"type": "category"},  # years are labels, not a numeric axis
        "yaxis": {"title": {"text": "return"}, "tickformat": ".0%"},
        "hovermode": "closest",
    }
    table = pd.DataFrame({"year": years, "return": values})
    return Section(
        "Returns by year", chart={"data": [trace], "layout": layout}, table=table, chart_height=260
    )


def exposure_section(result: BacktestResult) -> Section:
    e = result.equity
    idx = pd.DatetimeIndex(e["session_date"])
    traces = [
        _line(idx, e["gross_exposure"], "gross exposure", 0, hover="%{y:.0%}"),
        _line(idx, e["net_exposure"], "net exposure", 1, hover="%{y:.0%}"),
    ]
    table = e[["session_date", "gross_exposure", "net_exposure", "turnover", "n_positions"]].copy()
    table["session_date"] = pd.to_datetime(table["session_date"]).dt.date
    layout = {"yaxis": {"title": {"text": "share of equity"}, "tickformat": ".0%"}}
    return Section("Exposure", chart={"data": traces, "layout": layout}, table=table, chart_height=240)


def summary_tiles(result: BacktestResult) -> Section:
    m = result.metrics
    kv = {
        "Total return": _pct(m.get("total_return")),
        "CAGR": _pct(m.get("cagr")),
        "Volatility": _pct(m.get("volatility")),
        "Sharpe": _num(m.get("sharpe")),
        "Sortino": _num(m.get("sortino")),
        "Max drawdown": _pct(m.get("max_drawdown")),
        "Calmar": _num(m.get("calmar")),
        "Hit rate": _pct(m.get("hit_rate")),
        "Annual turnover": _num(m.get("annual_turnover"), "{:.1f}x"),
        "Sessions": str(m.get("sessions")),
        "Fills": str(len(result.fills)),
        "Final equity": _num(m.get("final_equity"), "{:,.0f}"),
    }
    for key in ("beta", "alpha_annual", "information_ratio"):
        if key in m:
            kv[key.replace("_", " ")] = _pct(m[key]) if key == "alpha_annual" else _num(m[key])
    text = (
        f"{result.strategy.get('name')} on {len(result.strategy.get('instruments', []))} instrument(s), "
        f"{m.get('start')} to {m.get('end')}, base currency {result.config.get('base_currency')}. "
        f"{result.variant_count} variant(s) of this strategy were run in the current process."
    )
    return Section("Summary", text=text, kv=kv)


def costs_section(result: BacktestResult) -> Section:
    base = result.config.get("base_currency")
    rows = [{"cost": k, f"total ({base})": v} for k, v in result.costs.items()]
    table = (
        pd.DataFrame(rows) if rows else pd.DataFrame({"cost": ["none recorded"], f"total ({base})": [0.0]})
    )
    fills = result.fills
    kv = {
        "Fills": str(len(fills)),
        "Gross traded": _num(fills["gross_notional"].sum() if len(fills) else 0.0, "{:,.0f}"),
        "Total costs": _num(sum(result.costs.values()), "{:,.2f}"),
        "Costs / traded": _pct(sum(result.costs.values()) / fills["gross_notional"].sum())
        if len(fills)
        else "n/a",
    }
    return Section("Costs and trading", kv=kv, table=table)


def robustness_sections(robustness: Mapping[str, Any]) -> list[Section]:
    out: list[Section] = []
    wf = robustness.get("walk_forward")
    if wf is not None:
        oos = wf.oos_metrics
        kv = {
            "Windows": str(oos.get("windows")),
            "OOS total return": _pct(oos.get("total_return")),
            "OOS Sharpe": _num(oos.get("sharpe")),
            "OOS max drawdown": _pct(oos.get("max_drawdown")),
            "Selection metric": wf.selection_metric,
        }
        table = wf.windows.copy()
        table["params"] = table["params"].map(str)
        out.append(
            Section(
                "Walk-forward (out of sample)",
                kv=kv,
                table=table,
                text="Parameters chosen on each training window, evaluated on the following unseen window.",
            )
        )
    sens = robustness.get("sensitivity")
    if sens is not None:
        summary = robustness.get("sensitivity_summary") or {}
        kv = {
            k.replace("_", " "): (_num(v, "{:.3g}") if isinstance(v, float) else str(v))
            for k, v in summary.items()
        }
        out.append(
            Section(
                "Parameter sensitivity",
                kv=kv,
                table=sens,
                text="The same strategy across a parameter grid; an edge found in one cell only is a fit.",
            )
        )
    boot = robustness.get("bootstrap")
    if boot is not None:
        kv = {
            "Sharpe (point)": _num(boot.get("point")),
            f"{int(boot.get('ci', 0.95) * 100)}% interval": (
                f"[{_num(boot.get('lower'))}, {_num(boot.get('upper'))}]"
            ),
            "P(Sharpe < 0)": _pct(boot.get("prob_negative")),
            "Bootstrap draws / block": f"{boot.get('n_boot')} / {boot.get('block')} sessions",
        }
        out.append(Section("Block-bootstrap Sharpe interval", kv=kv))
    dsr = robustness.get("deflated_sharpe")
    if dsr is not None:
        kv = {
            "Observed Sharpe (annual)": _num(dsr.get("sharpe_annual")),
            "Variants tried": str(dsr.get("n_trials")),
            "Expected max Sharpe of noise": _num(dsr.get("expected_max_sharpe_annual")),
            "PSR vs zero": _pct(dsr.get("psr_vs_zero")),
            "Deflated Sharpe": _pct(dsr.get("deflated_sharpe")),
            "Skew / kurtosis": f"{_num(dsr.get('skew'))} / {_num(dsr.get('kurtosis'))}",
        }
        p = dsr.get("deflated_sharpe")
        verdict = (
            "n/a"
            if p is None or not np.isfinite(p)
            else f"{p:.0%} probability that the true Sharpe exceeds what the best of "
            f"{dsr.get('n_trials')} noise "
            "trials would show. Below ~95% the result is not distinguishable from selection luck."
        )
        out.append(Section("Deflated Sharpe ratio", kv=kv, text=verdict))
    return out


def config_section(result: BacktestResult) -> Section:
    cfg = {k: v for k, v in result.config.items() if k != "cost_model"}
    cm = result.config.get("cost_model", {})
    rows = [{"setting": k, "value": str(v)} for k, v in cfg.items()]
    rows += [
        {"setting": f"cost_model.{k}", "value": str(v)} for k, v in cm.items() if k != "unverified_fee_rules"
    ]
    params = result.strategy.get("params") or {}
    rows += [{"setting": f"strategy.{k}", "value": str(v)} for k, v in params.items()]
    return Section("Configuration", table=pd.DataFrame(rows))


def build_tearsheet(
    result: BacktestResult,
    *,
    robustness: Mapping[str, Any] | None = None,
    title: str | None = None,
    path: Path | str | None = None,
    inline_plotly: bool = False,
) -> str:
    sections = [
        summary_tiles(result),
        equity_section(result),
        drawdown_section(result),
        yearly_section(result),
        exposure_section(result),
        costs_section(result),
    ]
    if robustness:
        sections += robustness_sections(robustness)
    sections.append(config_section(result))
    caveats = list(result.caveats)
    if not robustness or "deflated_sharpe" not in robustness:
        caveats.append(
            "No deflated-Sharpe or out-of-sample analysis attached: the headline Sharpe is in-sample."
        )
    m = result.metrics
    subtitle = (
        f"{m.get('start')} to {m.get('end')} · {result.strategy.get('name')} · "
        f"base {result.config.get('base_currency')}"
    )
    return render_report(
        title or f"Tearsheet: {result.strategy.get('name')}",
        sections,
        caveats,
        subtitle=subtitle,
        path=path,
        inline_js=plotly_js_source() if inline_plotly else None,
    )
