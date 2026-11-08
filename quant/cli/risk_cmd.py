"""`quant risk ...`: VaR family, stress windows and shocks, drawdowns, correlation structure, full report."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

risk_app = typer.Typer(no_args_is_help=True, help="Risk analytics from cached data.")

InstrumentsOpt = Annotated[str, typer.Option("--instruments", "-i", help="Comma-separated refs.")]
WeightsOpt = Annotated[str | None, typer.Option("--weights", "-w", help="ref=w,... (default equal).")]
StartOpt = Annotated[str, typer.Option("--start", "-s")]
EndOpt = Annotated[str | None, typer.Option("--end", "-e")]
BaseOpt = Annotated[str | None, typer.Option("--base", help="Convert prices to this currency.")]
NoRefreshOpt = Annotated[bool, typer.Option("--no-refresh", help="Never fetch; cached data only.")]
AlphaOpt = Annotated[float, typer.Option("--alpha", help="Confidence level, e.g. 0.99.")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


def _portfolio(svc, instruments, weights, start, end, base, no_refresh):  # noqa: ANN001,ANN202
    from quant.portfolio import returns_from_prices

    refs = [svc.resolve(r.strip()).instrument_id for r in instruments.split(",") if r.strip()]
    if not refs:
        raise typer.BadParameter("give at least one instrument")
    w: dict[str, float] = {}
    for part in filter(None, (weights or "").split(",")):
        if "=" not in part:
            raise typer.BadParameter(f"--weights expects ref=value pairs, got {part!r}")
        k, v = part.split("=", 1)
        w[svc.resolve(k.strip()).instrument_id] = float(v)
    if not w:
        w = {r: 1.0 / len(refs) for r in refs}
    missing = [r for r in refs if r not in w]
    if missing:
        raise typer.BadParameter(f"--weights gives no weight for {missing}")
    base_ccy = base or svc.settings.trading.base_currency
    end_d = end or date.today().isoformat()
    prices = svc.close_panel(refs, start, end_d, to_currency=base_ccy, refresh=False if no_refresh else None)
    if prices.empty:
        console.print("[yellow]no cached prices for these instruments; run `quant data fetch`[/yellow]")
        raise typer.Exit(code=1)
    prices = prices.dropna(how="any")
    rets = returns_from_prices(prices)
    if len(rets) < 60:
        console.print(f"[yellow]only {len(rets)} overlapping return observations; refusing[/yellow]")
        raise typer.Exit(code=1)
    return base_ccy, prices, rets, pd.Series(w).reindex(prices.columns).fillna(0.0)


@risk_app.command("var")
def var(
    instruments: InstrumentsOpt,
    weights: WeightsOpt = None,
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    alpha: AlphaOpt = 0.99,
    horizon: Annotated[int, typer.Option("--horizon", help="Days.")] = 1,
    window: Annotated[
        int, typer.Option("--window", help="Rolling VaR window for the Kupiec backtest.")
    ] = 250,
    base: BaseOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """VaR / CVaR by every method on the constant-weight portfolio, plus a Kupiec backtest of rolling VaR."""
    from quant.risk import kupiec_test, portfolio_returns, rolling_historical_var, var_table

    svc = _service(config_file)
    base_ccy, _, rets, w = _portfolio(svc, instruments, weights, start, end, base, no_refresh)
    port = portfolio_returns(rets, w)
    table = var_table(port, alpha, horizon)
    console.print(frame_table(table, title=f"VaR {alpha:.0%} / {horizon}d on {len(port)} days ({base_ccy})"))
    if len(port) > window + 50:
        k = kupiec_test(port, rolling_historical_var(port, window, alpha), alpha)
        verdict = "rejected" if k["reject_at_5pct"] else "not rejected"
        console.print(
            f"Kupiec backtest of rolling {window}d historical VaR: {k['exceptions']} exceptions vs "
            f"{k['expected']:.1f} expected in {k['n']} days; p={k['p_value']:.3f} ({verdict} at 5%)"
        )
    else:
        console.print(f"[yellow]sample too short for a {window}-day rolling backtest[/yellow]")
    console.print(
        "[yellow]- Estimates from this sample only; not a forecast of the next tail event.[/yellow]"
    )


@risk_app.command("stress")
def stress(
    instruments: InstrumentsOpt,
    weights: WeightsOpt = None,
    start: StartOpt = "2007-01-01",
    end: EndOpt = None,
    shock: Annotated[list[str] | None, typer.Option("--shock", help="name:REF=ret,REF=ret")] = None,
    base: BaseOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Historical stress windows (coverage reported) and optional hypothetical instant shocks."""
    from quant.risk import historical_stress, hypothetical_shocks

    svc = _service(config_file)
    base_ccy, prices, _, w = _portfolio(svc, instruments, weights, start, end, base, no_refresh)
    st = historical_stress(w, prices)
    cols = [
        "window",
        "start",
        "end",
        "sessions",
        "coverage",
        "usable",
        "portfolio_return",
        "worst_day",
        "max_drawdown",
    ]
    console.print(frame_table(st[cols], title=f"historical stress windows ({base_ccy})"))
    uncovered = st.loc[~st["usable"], "window"].tolist()
    if uncovered:
        console.print(f"[yellow]not covered by the price history: {', '.join(uncovered)}[/yellow]")
    if shock:
        scenarios: dict[str, dict[str, float]] = {}
        for item in shock:
            if ":" not in item:
                raise typer.BadParameter(f"--shock expects name:REF=ret,...; got {item!r}")
            name, body = item.split(":", 1)
            scenarios[name] = {}
            for part in filter(None, body.split(",")):
                k, v = part.split("=", 1)
                scenarios[name][svc.resolve(k.strip()).instrument_id] = float(v)
        console.print(frame_table(hypothetical_shocks(w, scenarios), title="hypothetical shocks"))


@risk_app.command("drawdown")
def drawdown(
    ref: Annotated[str | None, typer.Argument(help="Instrument ref (or use --backtest).")] = None,
    backtest: Annotated[
        Path | None, typer.Option("--backtest", help="A `quant backtest run --out` dir.")
    ] = None,
    start: StartOpt = "2015-01-01",
    end: EndOpt = None,
    top: Annotated[int, typer.Option("--top")] = 10,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Drawdown episodes and underwater statistics for an instrument or a saved backtest."""
    from quant.risk import calmar_ratio, conditional_drawdown_at_risk, drawdown_episodes, underwater_summary

    if backtest is not None:
        eq_df = pd.read_parquet(Path(backtest) / "equity.parquet")
        equity = pd.Series(eq_df["equity"].to_numpy(), index=pd.DatetimeIndex(eq_df["session_date"]))
        label = str(backtest)
    elif ref:
        svc = _service(config_file)
        end_d = end or date.today().isoformat()
        panel = svc.close_panel([ref], start, end_d, refresh=False if no_refresh else None)
        if panel.empty:
            console.print(f"[yellow]no cached bars for {ref}[/yellow]")
            raise typer.Exit(code=1)
        equity = panel.iloc[:, 0].dropna()
        label = ref
    else:
        raise typer.BadParameter("give an instrument ref or --backtest DIR")
    table = drawdown_episodes(equity, top=top)
    console.print(frame_table(table, title=f"drawdown episodes: {label}"))
    summary = underwater_summary(equity)
    summary["calmar"] = calmar_ratio(equity)
    summary["cdar_95"] = conditional_drawdown_at_risk(equity, 0.95)
    rows = [{"metric": k, "value": f"{v:.4g}"} for k, v in summary.items()]
    console.print(frame_table(pd.DataFrame(rows), title="underwater summary"))


@risk_app.command("correlation")
def correlation(
    instruments: InstrumentsOpt,
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    window: Annotated[int, typer.Option("--window", help="Rolling window for average correlation.")] = 63,
    base: BaseOpt = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Average and rolling correlation, calm-vs-stress breakdown and absorption ratio."""
    from quant.portfolio import sample_cov
    from quant.risk import average_correlation, correlation_breakdown, correlation_summary

    svc = _service(config_file)
    base_ccy, _, rets, w = _portfolio(svc, instruments, None, start, end, base, no_refresh)
    console.print(
        frame_table(
            rets.corr().round(3).rename_axis("asset").reset_index(), title=f"correlation ({base_ccy})"
        )
    )
    summary = correlation_summary(rets, sample_cov(rets).cov, w)
    rows = [{"metric": k, "value": f"{v:.4g}"} for k, v in summary.items()]
    console.print(
        frame_table(pd.DataFrame(rows), title="summary (equal weights for the diversification ratio)")
    )
    if rets.shape[1] >= 2:
        br = correlation_breakdown(rets)
        console.print(
            frame_table(
                br["difference"].round(3).rename_axis("asset").reset_index(), title="stress minus calm"
            )
        )
        roll = average_correlation(rets, window)
        assert isinstance(roll, pd.Series)
        tail = roll.dropna().tail(5).rename_axis("date").reset_index()
        console.print(frame_table(tail, title=f"rolling {window}d average correlation (last 5)"))


@risk_app.command("report")
def report(
    instruments: InstrumentsOpt,
    html: Annotated[Path, typer.Option("--html", help="Output file.")],
    weights: WeightsOpt = None,
    start: StartOpt = "2015-01-01",
    end: EndOpt = None,
    alpha: AlphaOpt = 0.99,
    horizon: Annotated[int, typer.Option("--horizon")] = 1,
    base: BaseOpt = None,
    inline_plotly: Annotated[
        bool, typer.Option("--inline-plotly", help="Embed plotly.js (offline).")
    ] = False,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Self-contained HTML risk report: VaR table + backtest, drawdowns, stress windows, correlations."""
    from quant.portfolio import ledoit_wolf_cov
    from quant.risk import build_risk_report

    svc = _service(config_file)
    base_ccy, prices, rets, w = _portfolio(svc, instruments, weights, start, end, base, no_refresh)
    build_risk_report(
        rets,
        w,
        prices=prices,
        cov=ledoit_wolf_cov(rets).cov,
        alpha=alpha,
        horizon=horizon,
        title="Risk report",
        subtitle=(
            f"{', '.join(prices.columns)} · {base_ccy} · "
            f"{prices.index[0].date()} to {prices.index[-1].date()}"
        ),
        path=html,
        inline_plotly=inline_plotly,
    )
    console.print(f"wrote {html}")
