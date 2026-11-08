"""`quant optimize ...`: portfolio weights from cached prices (min-variance, mean-variance, max-Sharpe,
risk parity, HRP, Black-Litterman), the efficient frontier, and Kelly-style position sizing."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

optimize_app = typer.Typer(no_args_is_help=True, help="Portfolio construction from cached data.")

METHODS = ("minvar", "mv", "maxsharpe", "rp", "invvol", "hrp", "bl")

InstrumentsOpt = Annotated[str, typer.Option("--instruments", "-i", help="Comma-separated refs.")]
StartOpt = Annotated[str, typer.Option("--start", "-s")]
EndOpt = Annotated[str | None, typer.Option("--end", "-e")]
CovOpt = Annotated[str, typer.Option("--cov", help="ledoit_wolf | sample | ewma")]
BaseOpt = Annotated[
    str | None, typer.Option("--base", help="Convert prices to this currency (default: config).")
]
NoRefreshOpt = Annotated[bool, typer.Option("--no-refresh", help="Never fetch; cached data only.")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


def _refs(instruments: str) -> list[str]:
    refs = [r.strip() for r in instruments.split(",") if r.strip()]
    if len(refs) < 2:
        raise typer.BadParameter("give at least two instruments")
    return refs


def _parse_kv(spec: str | None, what: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in filter(None, (spec or "").split(",")):
        if "=" not in part:
            raise typer.BadParameter(f"{what} expects ref=value pairs, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _parse_views(items: list[str] | None, resolve) -> list[Any]:  # noqa: ANN001
    """'AAPL:0.08@0.6' absolute view, 'AAPL-MSFT:0.02' relative view (optional @confidence)."""
    from quant.portfolio import View

    views = []
    for item in items or []:
        if ":" not in item:
            raise typer.BadParameter(f"--view expects REF[-REF]:return[@confidence], got {item!r}")
        lhs, rhs = item.split(":", 1)
        conf = None
        if "@" in rhs:
            rhs, c = rhs.split("@", 1)
            conf = float(c)
        picks: dict[str, float] = {}
        if "-" in lhs:
            a, b = lhs.split("-", 1)
            picks[resolve(a.strip())] = 1.0
            picks[resolve(b.strip())] = -1.0
        else:
            picks[resolve(lhs.strip())] = 1.0
        views.append(View(picks, float(rhs), conf))
    return views


def _inputs(svc, refs, start, end, base, cov_method, no_refresh):  # noqa: ANN001,ANN202
    from quant.portfolio import covariance, returns_from_prices

    base_ccy = base or svc.settings.trading.base_currency
    end_d = end or date.today().isoformat()
    panel = svc.close_panel(refs, start, end_d, to_currency=base_ccy, refresh=False if no_refresh else None)
    if panel.empty or panel.shape[1] < 2:
        console.print(
            "[yellow]need cached prices for at least two instruments; run `quant data fetch`[/yellow]"
        )
        raise typer.Exit(code=1)
    rets = returns_from_prices(panel.dropna(how="any"))
    if len(rets) < 60:
        console.print(
            f"[yellow]only {len(rets)} overlapping return observations; refusing to optimise[/yellow]"
        )
        raise typer.Exit(code=1)
    est = covariance(rets, cov_method)
    return base_ccy, panel, rets, est


def _expected_returns(rets: pd.DataFrame, shrink: float, periods_per_year: int = 252) -> pd.Series:
    """Annualised historical means shrunk toward their cross-sectional average (a weak, honest default)."""
    mu = rets.mean() * periods_per_year
    return ((1.0 - shrink) * mu + shrink * mu.mean()).rename("mu")


@optimize_app.command("weights")
def weights(
    instruments: InstrumentsOpt,
    method: Annotated[str, typer.Option("--method", "-m", help=" | ".join(METHODS))] = "hrp",
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    cov_method: CovOpt = "ledoit_wolf",
    base: BaseOpt = None,
    max_weight: Annotated[float, typer.Option("--max-weight")] = 1.0,
    long_short: Annotated[
        bool, typer.Option("--long-short", help="Allow negative weights (gross <= 2).")
    ] = False,
    risk_aversion: Annotated[float, typer.Option("--risk-aversion")] = 2.5,
    rf: Annotated[float, typer.Option("--rf", help="Annual risk-free rate for Sharpe.")] = 0.0,
    mu_shrink: Annotated[
        float, typer.Option("--mu-shrink", help="Shrink historical means to their average.")
    ] = 0.5,
    market_weights: Annotated[str | None, typer.Option("--market-weights", help="bl: ref=w,...")] = None,
    view: Annotated[list[str] | None, typer.Option("--view", help="bl: REF[-REF]:ret[@conf]")] = None,
    target_vol: Annotated[
        float | None, typer.Option("--target-vol", help="Scale to this ex-ante vol.")
    ] = None,
    max_leverage: Annotated[float, typer.Option("--max-leverage")] = 2.0,
    html: Annotated[Path | None, typer.Option("--html")] = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Optimise weights for a set of instruments from cached (adjusted) closes converted to one currency."""
    from quant.portfolio import (
        Constraints,
        OptimisationResult,
        bl_weights,
        black_litterman,
        hrp,
        inverse_volatility,
        max_sharpe,
        mean_variance,
        min_variance,
        risk_contributions,
        risk_parity,
        scale_to_target_vol,
    )

    if method not in METHODS:
        raise typer.BadParameter(f"unknown method {method!r}; choose from {METHODS}")
    svc = _service(config_file)
    refs = [svc.resolve(r).instrument_id for r in _refs(instruments)]
    base_ccy, panel, rets, est = _inputs(svc, refs, start, end, base, cov_method, no_refresh)
    cov = est.cov
    cons = Constraints(long_only=not long_short, upper=max_weight, max_gross=2.0 if long_short else None)
    caveats = [
        f"Covariance: {est.method} over {est.n_obs} daily observations ({panel.index[0].date()}.."
        f"{panel.index[-1].date()}), annualised; it describes that history, not the future.",
        "Transaction costs, taxes, lot sizes and liquidity are ignored here; backtest the weights before "
        "use.",
    ] + est.notes
    mu = None
    extra_tables: dict[str, pd.DataFrame] = {}
    if method == "minvar":
        res = min_variance(cov, cons)
    elif method == "rp":
        res = (
            risk_parity(cov)
            if cons.long_only and max_weight >= 1.0
            else risk_parity(cov, bounds=(0.0, max_weight))
        )
    elif method == "invvol":
        w = inverse_volatility(cov)
        res = OptimisationResult(
            w, float(np.sqrt(w @ cov.to_numpy() @ w)), None, None, "inverse_vol", True, "ok"
        )
    elif method == "hrp":
        res = hrp(cov)
    elif method in {"mv", "maxsharpe"}:
        mu = _expected_returns(rets, mu_shrink)
        caveats.append(
            f"Expected returns are historical means shrunk {mu_shrink:.0%} toward their average: a weak "
            "input, "
            "and the optimiser amplifies its error."
        )
        res = (
            mean_variance(mu, cov, risk_aversion, cons, rf)
            if method == "mv"
            else max_sharpe(mu, cov, rf, cons)
        )
    else:  # bl
        mw = _parse_kv(market_weights, "--market-weights")
        if mw:
            mkt = pd.Series({svc.resolve(k).instrument_id: v for k, v in mw.items()}).reindex(cov.index)
            if mkt.isna().any():
                raise typer.BadParameter("--market-weights must cover every instrument")
            mkt = mkt / mkt.sum()
        else:
            mkt = pd.Series(1.0 / len(cov), index=cov.index)
            caveats.append("No --market-weights given: equal weights used as the equilibrium portfolio.")
        bl = black_litterman(
            cov, mkt, _parse_views(view, lambda r: svc.resolve(r).instrument_id), risk_aversion=risk_aversion
        )
        mu = bl.posterior_mean
        caveats += bl.notes
        res = mean_variance(mu, cov, risk_aversion, cons, rf)
        extra_tables["black_litterman"] = pd.DataFrame(
            {
                "prior": bl.prior,
                "posterior": bl.posterior_mean,
                "unconstrained_w": bl_weights(bl.posterior_mean, cov, risk_aversion),
            }
        )
    table = pd.DataFrame({"weight": res.weights, "risk_contribution": risk_contributions(res.weights, cov)})
    table["vol"] = est.vol()
    if mu is not None:
        table["mu"] = mu
    leverage = None
    if target_vol:
        scaled, leverage = scale_to_target_vol(res.weights, cov, target_vol, max_leverage)
        table["scaled_weight"] = scaled
        caveats.append(
            f"Scaled to {target_vol:.1%} ex-ante vol with leverage {leverage:.2f} (cap {max_leverage:g}); "
            "realised vol will differ."
        )
    console.print(
        frame_table(table.rename_axis("instrument").reset_index(), title=f"{res.method} ({base_ccy})")
    )
    status = "ok" if res.success else "[red]did not converge[/red]"
    er = (
        f"  E[r] {res.expected_return:.2%}  Sharpe {res.sharpe:.2f}"
        if res.expected_return is not None
        else ""
    )
    console.print(f"ex-ante vol {res.volatility:.2%}{er}  status {status}: {res.message}")
    if res.binding:
        console.print(f"binding constraints: {', '.join(res.binding)}")
    for name, frame in extra_tables.items():
        console.print(frame_table(frame.rename_axis("instrument").reset_index(), title=name))
    for c in caveats:
        console.print(f"[yellow]- {c}[/yellow]")
    if html:
        from quant.reports import Section, render_report

        sections = [
            Section(
                "Weights",
                kv={
                    "method": res.method,
                    "ex-ante vol": f"{res.volatility:.2%}",
                    "leverage": f"{leverage:.2f}" if leverage is not None else "n/a",
                    "binding": ", ".join(res.binding) or "none",
                },
                table=table.rename_axis("instrument").reset_index(),
            ),
            Section("Correlation", table=est.corr().round(3).rename_axis("instrument").reset_index()),
        ] + [Section(n, table=f.rename_axis("instrument").reset_index()) for n, f in extra_tables.items()]
        render_report(f"Portfolio weights ({res.method})", sections, caveats, subtitle=base_ccy, path=html)
        console.print(f"wrote {html}")


@optimize_app.command("frontier")
def frontier(
    instruments: InstrumentsOpt,
    start: StartOpt = "2018-01-01",
    end: EndOpt = None,
    cov_method: CovOpt = "ledoit_wolf",
    base: BaseOpt = None,
    points: Annotated[int, typer.Option("--points")] = 12,
    max_weight: Annotated[float, typer.Option("--max-weight")] = 1.0,
    rf: Annotated[float, typer.Option("--rf")] = 0.0,
    mu_shrink: Annotated[float, typer.Option("--mu-shrink")] = 0.5,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Long-only efficient frontier from shrunk historical means and the chosen covariance estimator."""
    from quant.portfolio import Constraints, efficient_frontier

    svc = _service(config_file)
    refs = [svc.resolve(r).instrument_id for r in _refs(instruments)]
    base_ccy, _, rets, est = _inputs(svc, refs, start, end, base, cov_method, no_refresh)
    mu = _expected_returns(rets, mu_shrink)
    fr = efficient_frontier(mu, est.cov, points, Constraints(upper=max_weight), rf)
    show = fr[["target_return", "expected_return", "volatility", "sharpe", "success"]].copy()
    console.print(frame_table(show, title=f"efficient frontier ({base_ccy}, {est.method})"))
    console.print(
        "[yellow]- Inputs are historical; the frontier describes the sample, not the future.[/yellow]"
    )


@optimize_app.command("size")
def size(
    win_rate: Annotated[float, typer.Option("--win-rate")],
    payoff: Annotated[float, typer.Option("--payoff", help="Average win / average loss, in R.")],
    entry: Annotated[float, typer.Option("--entry")],
    stop: Annotated[float, typer.Option("--stop")],
    capital: Annotated[float, typer.Option("--capital")],
    trades: Annotated[int, typer.Option("--trades", help="Sample size behind the win rate.")] = 0,
    cash: Annotated[float | None, typer.Option("--cash")] = None,
    invested: Annotated[float, typer.Option("--invested")] = 0.0,
    kelly: Annotated[float, typer.Option("--kelly", help="Fraction of full Kelly.")] = 0.25,
    max_risk_pct: Annotated[float, typer.Option("--max-risk-pct")] = 1.0,
    max_position_value: Annotated[float | None, typer.Option("--max-position-value")] = None,
    cash_reserve: Annotated[float, typer.Option("--cash-reserve")] = 0.0,
    regime: Annotated[str, typer.Option("--regime", help="risk_on | caution | risk_off")] = "risk_on",
    cost_bps: Annotated[float, typer.Option("--cost-bps", help="Round-trip cost in bps of entry.")] = 0.0,
    lot: Annotated[int, typer.Option("--lot")] = 1,
    target: Annotated[float | None, typer.Option("--target")] = None,
) -> None:
    """Fractional-Kelly position size with Wilson shrinkage and ordered caps (long trades, one currency)."""
    from quant.portfolio import size_position

    res = size_position(
        win_rate=win_rate,
        payoff=payoff,
        entry=entry,
        stop=stop,
        capital=capital,
        n_trades=trades,
        cash=cash,
        invested=invested,
        kelly_multiplier=kelly,
        max_risk_per_trade_pct=max_risk_pct,
        max_position_value=max_position_value,
        cash_reserve=cash_reserve,
        regime=regime,
        round_trip_cost_bps=cost_bps,
        target=target,
        lot_size=lot,
    )
    console.print(
        f"[bold]{res.decision}[/bold] qty {res.qty:g}  value {res.position_value:,.2f}  "
        f"at risk {res.amount_at_risk:,.2f} ({(res.risk_pct_of_capital or 0):.2f}% of capital)  "
        f"binding: {res.binding_constraint}"
    )
    console.print(
        f"Kelly full {res.kelly_full_raw:.3f} -> conservative {res.kelly_full_conservative:.3f} "
        f"-> used {res.kelly_used:.4f}"
    )
    if res.caps_qty:
        console.print("caps: " + ", ".join(f"{k}={v:g}" for k, v in res.caps_qty.items()))
    for n in res.notes:
        console.print(f"[yellow]- {n}[/yellow]")
