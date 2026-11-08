"""`quant fundamentals ...`: statements, ratios, DCF, comparables scores and earnings surprises."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

fundamentals_app = typer.Typer(
    no_args_is_help=True, help="Point-in-time fundamentals, valuation and earnings."
)
AsOfOpt = Annotated[str | None, typer.Option("--as-of", help="Only statements filed by this date.")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


def _kv(d: dict, title: str) -> None:
    rows = [
        {"name": k, "value": (f"{v:,.4g}" if isinstance(v, float) else str(v))}
        for k, v in d.items()
        if v is not None
    ]
    console.print(frame_table(pd.DataFrame(rows), title=title, max_rows=200))


@fundamentals_app.command("fetch")
def fetch(
    refs: Annotated[list[str], typer.Argument()],
    vendor: Annotated[str | None, typer.Option("--vendor", help="yfinance | sec_edgar")] = None,
    earnings: Annotated[bool, typer.Option("--earnings", help="Also fetch earnings dates.")] = True,
    config_file: ConfigOption = None,
) -> None:
    """Download statements (and earnings dates) into the cache."""
    svc = _service(config_file)
    rep = svc.fetch_fundamentals(refs, vendor=vendor)
    console.print(f"{rep.vendor}: {rep.rows} statement rows; errors: {rep.errors or 'none'}")
    if earnings:
        er = svc.fetch_earnings(refs)
        console.print(f"{er.vendor}: {er.rows} earnings rows; errors: {er.errors or 'none'}")
    if rep.errors and rep.rows == 0:
        raise typer.Exit(code=1)


@fundamentals_app.command("ratios")
def ratios(
    ref: Annotated[str, typer.Argument()],
    as_of: AsOfOpt = None,
    price: Annotated[float | None, typer.Option("--price", help="Override the cached last close.")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Margins, returns on capital, leverage, growth and market multiples at a point in time."""
    from quant.fundamental import compute_ratios

    svc = _service(config_file)
    stmts = svc.get_fundamentals(ref, as_of=as_of, refresh=False)
    if stmts.rows.empty:
        console.print(
            f"[yellow]no cached statements for {ref}; run `quant fundamentals fetch {ref}`[/yellow]"
        )
        raise typer.Exit(code=1)
    px = price if price is not None else svc.last_close(ref, as_of)
    r = compute_ratios(stmts, price=px, as_of=as_of)
    r.pop("assumptions", None)
    _kv(r, f"{ref} ratios" + (f" as of {as_of}" if as_of else ""))


@fundamentals_app.command("dcf")
def dcf(
    ref: Annotated[str, typer.Argument()],
    growth: Annotated[float, typer.Option("--growth", help="Explicit-period FCF growth per year.")] = 0.05,
    years: Annotated[int, typer.Option("--years")] = 5,
    terminal: Annotated[float, typer.Option("--terminal", help="Terminal growth.")] = 0.02,
    rate: Annotated[
        float | None, typer.Option("--rate", help="Discount rate (overrides WACC inputs).")
    ] = None,
    rf: Annotated[float, typer.Option("--rf")] = 0.04,
    erp: Annotated[float, typer.Option("--erp", help="Equity risk premium.")] = 0.05,
    beta: Annotated[float, typer.Option("--beta")] = 1.0,
    cost_of_debt: Annotated[float, typer.Option("--cost-of-debt")] = 0.05,
    debt_weight: Annotated[float, typer.Option("--debt-weight")] = 0.2,
    fcf: Annotated[float | None, typer.Option("--fcf", help="Override TTM free cash flow.")] = None,
    as_of: AsOfOpt = None,
    config_file: ConfigOption = None,
) -> None:
    """FCFF discounted-cash-flow value per share with a discount-rate x terminal-growth sensitivity grid."""
    from quant.fundamental import compute_ratios, dcf_sensitivity, dcf_valuation, wacc

    svc = _service(config_file)
    stmts = svc.get_fundamentals(ref, as_of=as_of, refresh=False)
    if stmts.rows.empty:
        console.print(
            f"[yellow]no cached statements for {ref}; run `quant fundamentals fetch {ref}`[/yellow]"
        )
        raise typer.Exit(code=1)
    r = compute_ratios(stmts, price=svc.last_close(ref, as_of), as_of=as_of)
    fcf0 = fcf if fcf is not None else r.get("free_cash_flow_ttm")
    if fcf0 is None:
        console.print("[red]no free cash flow available; pass --fcf[/red]")
        raise typer.Exit(code=1)
    discount = (
        rate
        if rate is not None
        else wacc(rf, erp, beta, cost_of_debt, r.get("tax_rate") or 0.21, debt_weight)
    )
    res = dcf_valuation(
        fcf0,
        growth,
        years=years,
        terminal_growth=terminal,
        discount_rate=discount,
        net_debt=r.get("net_debt") or 0.0,
        shares=r.get("shares"),
    )
    out = res.as_dict()
    out["market_price"] = r.get("price")
    _kv(out, f"{ref} DCF")
    grid = dcf_sensitivity(
        fcf0,
        growth,
        discount_rates=[discount - 0.02, discount - 0.01, discount, discount + 0.01, discount + 0.02],
        terminal_growths=[terminal - 0.01, terminal, terminal + 0.01],
        years=years,
        net_debt=r.get("net_debt") or 0.0,
        shares=r.get("shares"),
    )
    console.print(
        frame_table(
            grid.reset_index(), title="per-share value: discount rate (rows) x terminal growth (cols)"
        )
    )
    console.print(
        "[dim]A DCF is an assumption machine: the grid shows how much two of them move the answer.[/dim]"
    )


@fundamentals_app.command("scores")
def scores(
    refs: Annotated[list[str], typer.Argument(help="Instrument refs, or one stored universe name.")],
    as_of: AsOfOpt = None,
    config_file: ConfigOption = None,
) -> None:
    """Cross-sectional quality / value / growth percentile scores (descriptive, not predictive)."""
    from quant.fundamental import RATIO_KEYS, compute_ratios, score_cross_section

    svc = _service(config_file)
    if len(refs) == 1:
        members = svc.universe(refs[0]).membership_at(as_of or date.today())
        if members:
            refs = members
    rows = {}
    for ref in refs:
        stmts = svc.get_fundamentals(ref, as_of=as_of, refresh=False)
        if stmts.rows.empty:
            continue
        r = compute_ratios(stmts, price=svc.last_close(ref, as_of), as_of=as_of)
        rows[stmts.instrument_id] = {k: r.get(k) for k in RATIO_KEYS}
    if len(rows) < 2:
        console.print("[yellow]need cached fundamentals for at least two instruments[/yellow]")
        raise typer.Exit(code=1)
    table = score_cross_section(pd.DataFrame.from_dict(rows, orient="index"))
    console.print(
        frame_table(
            table.rename_axis("instrument_id").reset_index(), title=f"scores ({len(table)} instruments)"
        )
    )


@fundamentals_app.command("earnings")
def earnings(
    ref: Annotated[str, typer.Argument()],
    as_of: AsOfOpt = None,
    config_file: ConfigOption = None,
) -> None:
    """Earnings surprises, SUE and the beat rate."""
    from quant.fundamental import earnings_summary, surprises

    svc = _service(config_file)
    df = svc.get_earnings(ref, as_of=date.fromisoformat(as_of) if as_of else None, refresh=False)
    if df.empty:
        console.print(f"[yellow]no cached earnings for {ref}; run `quant fundamentals fetch {ref}`[/yellow]")
        raise typer.Exit(code=1)
    s = surprises(df)
    console.print(
        frame_table(
            s[["report_date", "eps_estimate", "eps_actual", "surprise", "surprise_pct", "sue", "beat"]].tail(
                12
            ),
            title=f"{ref} earnings",
        )
    )
    _kv(earnings_summary(s), "summary")
