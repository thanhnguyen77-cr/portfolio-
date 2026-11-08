"""`quant factors ...`: build long-short factor returns, measure exposures, load Ken French data."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

factors_app = typer.Typer(no_args_is_help=True, help="Factor returns, exposures and attribution.")


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


@factors_app.command("build")
def build(
    name: Annotated[str, typer.Argument(help="Name to save the factor returns under.")],
    universe: Annotated[str | None, typer.Option("--universe", "-u")] = None,
    instruments: Annotated[
        str | None, typer.Option("--instruments", "-i", help="Comma-separated refs.")
    ] = None,
    start: Annotated[str, typer.Option("--start", "-s")] = "2018-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    factors: Annotated[
        str, typer.Option("--factors", help="Comma-separated subset.")
    ] = "momentum,low_vol,size,value,quality",
    quantiles: Annotated[int, typer.Option("--quantiles")] = 5,
    sector_neutral: Annotated[bool, typer.Option("--sector-neutral")] = False,
    no_refresh: Annotated[bool, typer.Option("--no-refresh")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Dollar-neutral top-minus-bottom quantile factor returns, rebalanced monthly, from cached data."""
    svc = _service(config_file)
    end_d = end or date.today().isoformat()
    if universe:
        refs = svc.universe(universe).membership_at(end_d)
        if not refs:
            console.print(f"[yellow]universe {universe} is empty[/yellow]")
            raise typer.Exit(code=1)
    elif instruments:
        refs = [r.strip() for r in instruments.split(",") if r.strip()]
    else:
        console.print("[red]give --universe or --instruments[/red]")
        raise typer.Exit(code=2)
    fr = svc.build_factor_returns(
        refs,
        start,
        end_d,
        factors=[f.strip() for f in factors.split(",")],
        n_quantiles=quantiles,
        sector_neutral=sector_neutral,
        refresh=False if no_refresh else None,
    )
    if fr.returns.empty or fr.returns.shape[1] == 0:
        console.print("[yellow]no factor returns could be built[/yellow]")
        for n in fr.notes:
            console.print(f"  - {n}")
        raise typer.Exit(code=1)
    path = svc.save_factor_returns(name, fr.returns, fr.notes)
    console.print(
        frame_table(fr.stats().reset_index(), title=f"factor returns '{name}' ({len(fr.returns)} sessions)")
    )
    for n in fr.notes:
        console.print(f"[yellow]- {n}[/yellow]")
    console.print(f"saved {path}")


@factors_app.command("exposures")
def exposures(
    ref: Annotated[str, typer.Argument(help="Instrument ref, or a backtest --out directory.")],
    factors: Annotated[str, typer.Option("--factors", "-f", help="Saved factor-return name.")] = ...,
    start: Annotated[str, typer.Option("--start", "-s")] = "2018-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    rf: Annotated[float | None, typer.Option("--rf", help="Annual risk-free rate to subtract.")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Alpha, betas (HAC t-stats), R-squared and an arithmetic return attribution on saved factors."""
    from pathlib import Path

    from quant.factors import attribute_returns, factor_exposures
    from quant.factors.attribution import attribution_summary

    svc = _service(config_file)
    fac = svc.load_factor_returns(factors)
    path = Path(ref)
    if path.is_dir() and (path / "equity.parquet").exists():
        eq = pd.read_parquet(path / "equity.parquet")
        series = pd.Series(
            eq["equity"].to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(eq["session_date"]))
        )
        label = path.name
    else:
        bars = svc.get_adjusted_bars([ref], start, end or date.today().isoformat(), refresh=False)
        if bars.empty:
            console.print(f"[yellow]no cached bars for {ref}[/yellow]")
            raise typer.Exit(code=1)
        series = bars.set_index("session_date")["adj_close"]
        label = ref
    rets = series.pct_change().dropna()
    res = factor_exposures(rets, fac, rf=rf)
    console.print(
        frame_table(
            res.table().rename_axis("factor").reset_index(),
            title=f"{label} exposures ({res.cov_type}, n={res.nobs})",
        )
    )
    console.print(f"alpha {res.alpha_annual:+.2%} per year (t={res.alpha_t:.2f}), R² {res.r2:.3f}")
    contrib = attribute_returns(rets, fac, res, rf=rf)
    console.print(frame_table(attribution_summary(contrib).reset_index(), title="arithmetic attribution"))


@factors_app.command("kenfrench")
def kenfrench(
    key: Annotated[str, typer.Option("--file", help="ff3_daily | ff5_daily | momentum_daily")] = "ff5_daily",
    config_file: ConfigOption = None,
) -> None:
    """Download a Ken French daily factor file (US) and save it as factor returns named kenfrench_<file>."""
    from quant.factors import load_ken_french

    svc = _service(config_file)
    try:
        df = load_ken_french(key)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]download failed: {exc}[/red]")
        raise typer.Exit(code=1) from None
    path = svc.save_factor_returns(
        f"kenfrench_{key}", df, ["Source: Ken French data library (US, percent converted to decimal)"]
    )
    console.print(f"saved {len(df)} rows to {path}")
