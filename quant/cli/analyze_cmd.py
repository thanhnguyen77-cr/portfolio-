"""`quant analyze ...`: technical and statistical analysis of cached series (adjusted prices)."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

analyze_app = typer.Typer(no_args_is_help=True, help="Technical and statistical analysis of cached series.")

StartOpt = Annotated[str, typer.Option("--start", "-s")]
EndOpt = Annotated[str | None, typer.Option("--end", "-e")]
NoRefreshOpt = Annotated[bool, typer.Option("--no-refresh", help="Use cached bars only.")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


def _series(svc, ref: str, start: str, end: str | None, no_refresh: bool) -> pd.DataFrame:  # noqa: ANN001
    end_d = end or date.today().isoformat()
    bars = svc.get_adjusted_bars([ref], start, end_d, refresh=False if no_refresh else None)
    if bars.empty:
        console.print(f"[yellow]no bars for {ref}[/yellow]")
        raise typer.Exit(code=1)
    df = bars.set_index("session_date")[["adj_open", "adj_high", "adj_low", "adj_close", "volume"]]
    return df.rename(columns=lambda c: c.replace("adj_", ""))


def _kv(d: dict, title: str) -> None:
    rows = [{"name": k, "value": (f"{v:.6g}" if isinstance(v, float) else str(v))} for k, v in d.items()]
    console.print(frame_table(pd.DataFrame(rows), title=title, max_rows=200))


@analyze_app.command("indicators")
def indicators(
    ref: Annotated[str, typer.Argument()],
    start: StartOpt = "2020-01-01",
    end: EndOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Latest trend / momentum / volatility / volume indicator values and regime labels."""
    from quant.technical import indicator_frame, trend_regime, volatility_regime

    df = _series(_service(config_file), ref, start, end, no_refresh)
    ind = indicator_frame(df)
    last = ind.iloc[-1].dropna()
    _kv({k: float(v) for k, v in last.items()}, f"{ref} indicators @ {ind.index[-1].date()}")
    tr = trend_regime(df["close"], high=df["high"], low=df["low"]).iloc[-1]
    vr = volatility_regime(df["close"]).iloc[-1]
    _kv(
        {
            "trend_regime": tr["trend_regime"],
            "adx": float(tr.get("adx", np.nan)),
            "vol_regime": vr["vol_regime"],
            "vol_percentile": float(vr["vol_percentile"]) if pd.notna(vr["vol_percentile"]) else float("nan"),
        },
        "regimes",
    )


@analyze_app.command("levels")
def levels(
    ref: Annotated[str, typer.Argument()],
    start: StartOpt = "2022-01-01",
    end: EndOpt = None,
    tolerance: Annotated[
        float, typer.Option("--tolerance", help="Cluster width as a fraction of price.")
    ] = 0.01,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Support and resistance levels from clustered swing pivots."""
    from quant.technical import nearest_levels, support_resistance

    df = _series(_service(config_file), ref, start, end, no_refresh)
    table = support_resistance(df["high"], df["low"], df["close"], tolerance=tolerance)
    if table.empty:
        console.print("[yellow]no levels with enough touches[/yellow]")
        raise typer.Exit(code=1)
    console.print(frame_table(table, title=f"{ref} levels (last close {df['close'].iloc[-1]:.4g})"))
    _kv(nearest_levels(table, float(df["close"].iloc[-1])), "nearest")


@analyze_app.command("regime")
def regime(
    ref: Annotated[str, typer.Argument(help="Benchmark, e.g. SPX or UKX.")],
    vol_index: Annotated[
        str | None, typer.Option("--vol-index", help="Volatility index ref, e.g. VIX.")
    ] = None,
    markov: Annotated[
        bool, typer.Option("--markov", help="Also fit a 2-state Markov-switching model.")
    ] = False,
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Rule-based risk regime (risk_on / caution / risk_off) and, optionally, Markov-switching regimes."""
    from quant.technical import markov_regimes, rule_based_regime

    svc = _service(config_file)
    df = _series(svc, ref, start, end, no_refresh)
    vi = _series(svc, vol_index, start, end, no_refresh)["close"] if vol_index else None
    rb = rule_based_regime(df["close"], vi).dropna(subset=["regime"])
    console.print(frame_table(rb.tail(10).reset_index(), title=f"{ref} rule-based regime (last 10 sessions)"))
    if markov:
        mk = markov_regimes(df["close"].pct_change().dropna())
        _kv(
            {
                "means_daily": mk.means,
                "vol_annual": [float(np.sqrt(v * 252)) for v in mk.variances],
                "transition": mk.transition.round(3).tolist(),
                "converged": mk.converged,
                "p_calm_now": float(mk.filtered.iloc[-1, 0]),
                "p_turbulent_now": float(mk.filtered.iloc[-1, -1]),
            },
            "markov",
        )


@analyze_app.command("stationarity")
def stationarity(
    ref: Annotated[str, typer.Argument()],
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """ADF, KPSS, Phillips-Perron and variance-ratio tests on log prices and on returns."""
    from quant.stats import stationarity_summary

    df = _series(_service(config_file), ref, start, end, no_refresh)
    rows = []
    for label, series in (("log_price", np.log(df["close"])), ("returns", df["close"].pct_change().dropna())):
        summary = stationarity_summary(series)
        for test, res in summary.items():
            if test == "verdict":
                continue
            rows.append(
                {
                    "series": label,
                    "test": test,
                    "statistic": res["statistic"],
                    "pvalue": res["pvalue"],
                    "null": res["null"],
                    "reject_5pct": res["reject_at_5pct"],
                }
            )
        rows.append(
            {
                "series": label,
                "test": "verdict",
                "statistic": np.nan,
                "pvalue": np.nan,
                "null": "",
                "reject_5pct": summary["verdict"],
            }
        )
    console.print(frame_table(pd.DataFrame(rows), title=f"{ref} stationarity"))


@analyze_app.command("cointegration")
def cointegration(
    ref_y: Annotated[str, typer.Argument()],
    ref_x: Annotated[str, typer.Argument()],
    start: StartOpt = "2020-01-01",
    end: EndOpt = None,
    z_window: Annotated[int, typer.Option("--z-window")] = 60,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Engle-Granger and Johansen tests, hedge ratio, half-life and the current spread z-score."""
    from quant.stats import engle_granger, johansen, spread_zscore

    svc = _service(config_file)
    y = _series(svc, ref_y, start, end, no_refresh)["close"]
    x = _series(svc, ref_x, start, end, no_refresh)["close"]
    eg = engle_granger(y, x)
    jo = johansen(pd.concat([y.rename(ref_y), x.rename(ref_x)], axis=1))
    z = spread_zscore(eg.spread, z_window).dropna()
    _kv(
        {
            "eg_statistic": eg.statistic,
            "eg_pvalue": eg.pvalue,
            "cointegrated_5pct": eg.cointegrated_at_5pct,
            "hedge_ratio": eg.hedge_ratio,
            "intercept": eg.intercept,
            "half_life_sessions": eg.half_life_sessions,
            "johansen_rank_95": jo.rank_at_95,
            "johansen_trace": [round(v, 2) for v in jo.trace_stats],
            "johansen_crit_95": [round(v, 2) for v in jo.trace_crit_95],
            "spread_z_now": float(z.iloc[-1]) if len(z) else float("nan"),
        },
        f"{ref_y} ~ {ref_x}",
    )


@analyze_app.command("garch")
def garch(
    ref: Annotated[str, typer.Argument()],
    model: Annotated[str, typer.Option("--model", help="GARCH | GJR | EGARCH")] = "GARCH",
    dist: Annotated[str, typer.Option("--dist", help="normal | t | skewt")] = "t",
    horizon: Annotated[int, typer.Option("--horizon")] = 10,
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Fit a GARCH-family model to daily returns and forecast annualised volatility."""
    from quant.stats import fit_garch, forecast_vol

    df = _series(_service(config_file), ref, start, end, no_refresh)
    res = fit_garch(df["close"].pct_change().dropna(), model=model, dist=dist)
    _kv(
        {
            **res.params,
            "persistence": res.persistence,
            "loglik": res.loglik,
            "aic": res.aic,
            "converged": res.converged,
            "cond_vol_now_annual": float(res.conditional_vol.iloc[-1]),
        },
        f"{ref} {res.model}({dist})",
    )
    fc = forecast_vol(res, horizon)
    console.print(frame_table(fc.rename_axis("step").reset_index(), title="annualised vol forecast"))


@analyze_app.command("pca")
def pca_cmd(
    refs: Annotated[list[str], typer.Argument()],
    start: StartOpt = "2020-01-01",
    end: EndOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Principal components of the return panel and the absorption ratio."""
    from quant.stats import absorption_ratio, pca

    svc = _service(config_file)
    panel = pd.concat([_series(svc, r, start, end, no_refresh)["close"].rename(r) for r in refs], axis=1)
    rets = panel.pct_change().dropna(how="any")
    res = pca(rets)
    console.print(
        frame_table(
            res.explained_variance_ratio.rename_axis("component").reset_index(), title="explained variance"
        )
    )
    console.print(frame_table(res.loadings.rename_axis("asset").reset_index(), title="loadings"))
    _kv({"absorption_ratio": float(absorption_ratio(rets)), "observations": int(len(rets))}, "absorption")
