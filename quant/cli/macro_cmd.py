"""`quant macro ...`: catalogue series, yield curve, inflation and the cross-asset regime board."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

macro_app = typer.Typer(no_args_is_help=True, help="Rates, inflation, yield curve and cross-asset regime.")
RegionOpt = Annotated[str, typer.Option("--region", "-r", help="us | uk")]
NoRefreshOpt = Annotated[bool, typer.Option("--no-refresh", help="Use cached observations only.")]


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


@macro_app.command("catalog")
def catalog_cmd(region: Annotated[str, typer.Option("--region", "-r")] = "both") -> None:
    """List the named macro series (vendor, id, frequency, verified flag)."""
    from quant.macro.catalog import series_for_region

    rows = [s.__dict__ for s in series_for_region(region)]
    console.print(frame_table(pd.DataFrame(rows), title=f"{len(rows)} macro series", max_rows=200))


@macro_app.command("series")
def series(
    name: Annotated[str, typer.Argument(help="Catalogue name, e.g. us_cpi or uk_bank_rate.")],
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    tail: Annotated[int, typer.Option("--tail", "-n")] = 12,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Fetch (incrementally) and print a catalogue series."""
    svc = _service(config_file)
    s = svc.macro_series(name, start, end, refresh=False if no_refresh else None)
    if s.empty:
        console.print(f"[yellow]no observations for {name}[/yellow]")
        raise typer.Exit(code=1)
    console.print(
        frame_table(s.tail(tail).rename_axis("date").reset_index(), title=f"{name} ({len(s)} observations)")
    )


@macro_app.command("curve")
def curve(
    region: RegionOpt = "us",
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Tenor snapshot, 10y-2y (US) or 10y-Bank Rate (UK) slope and inversion flags."""
    from quant.macro import curve_snapshot, inversion_flags, slope

    svc = _service(config_file)
    refresh = False if no_refresh else None
    names = {
        "us": ["us_3m", "us_2y", "us_5y", "us_10y", "us_30y"],
        "uk": ["uk_bank_rate", "uk_5y", "uk_10y", "uk_20y"],
    }[region]
    tenors = {}
    for n in names:
        try:
            s = svc.macro_series(n, start, end, refresh=refresh)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]{n}: {exc}[/yellow]")
            continue
        if not s.empty:
            tenors[n] = s
    if not tenors:
        console.print("[yellow]no cached tenor series; run `quant macro series us_10y` etc. first[/yellow]")
        raise typer.Exit(code=1)
    console.print(frame_table(curve_snapshot(tenors), title=f"{region} curve snapshot"))
    long_key, short_key = ("us_10y", "us_2y") if region == "us" else ("uk_10y", "uk_bank_rate")
    if long_key in tenors and short_key in tenors:
        flags = inversion_flags(slope(tenors[long_key], tenors[short_key]))
        console.print(
            frame_table(flags.tail(10).rename_axis("date").reset_index(), title=f"{long_key} - {short_key}")
        )
        inverted_now = bool(flags["inverted"].iloc[-1])
        n_inverted = int(flags["sessions_inverted"].iloc[-1])
        console.print(f"inverted now: {inverted_now}; sessions inverted: {n_inverted}")


@macro_app.command("inflation")
def inflation(
    region: RegionOpt = "us",
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    no_refresh: NoRefreshOpt = False,
    config_file: ConfigOption = None,
) -> None:
    """Year-over-year, 3-month annualised impulse and the inflation regime label."""
    from quant.macro import annualised_3m, inflation_regime, yoy

    svc = _service(config_file)
    refresh = False if no_refresh else None
    if region == "us":
        cpi = svc.macro_series("us_cpi", start, end, refresh=refresh)
        if cpi.empty:
            console.print("[yellow]no cached us_cpi[/yellow]")
            raise typer.Exit(code=1)
        table = pd.concat([yoy(cpi), annualised_3m(cpi)], axis=1)
        regime = inflation_regime(table["yoy"])
    else:
        rate = svc.macro_series("uk_cpih", start, end, refresh=refresh)
        if rate.empty:
            console.print("[yellow]no cached uk_cpih[/yellow]")
            raise typer.Exit(code=1)
        table = rate.to_frame("yoy")
        regime = inflation_regime(rate)
    table = table.join(regime[["regime"]])
    console.print(
        frame_table(table.dropna().tail(12).rename_axis("date").reset_index(), title=f"{region} inflation")
    )


@macro_app.command("dashboard")
def dashboard(
    region: RegionOpt = "us",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    html: Annotated[Path | None, typer.Option("--html")] = None,
    no_refresh: NoRefreshOpt = True,
    config_file: ConfigOption = None,
) -> None:
    """Cross-asset regime board: growth, inflation, rates, credit, FX, equities, volatility."""
    from quant.reports import Section, render_report

    svc = _service(config_file)
    board = svc.macro_dashboard(region, end=end, refresh=False if no_refresh else None)
    table = board["table"]
    if table.empty:
        console.print("[yellow]no inputs available; fetch macro series and index bars first[/yellow]")
        for n in board["notes"]:
            console.print(f"  - {n}")
        raise typer.Exit(code=1)
    console.print(frame_table(table, title=f"{region} cross-asset board @ {board['as_of']}"))
    console.print(
        f"growth x inflation quadrant: [bold]{board['quadrant']}[/bold]; signals: {board['risk_signals']}"
    )
    if board["missing"]:
        console.print(f"[yellow]missing inputs: {', '.join(board['missing'])}[/yellow]")
    for n in board["notes"]:
        console.print(f"[yellow]- {n}[/yellow]")
    if html:
        caveats = [
            "Quadrant uses the direction of 3-month changes in growth and inflation proxies, not forecasts.",
            "Z-scores are versus the trailing five years of each series; signals are heuristics, not advice.",
            f"Missing inputs: {board['missing'] or 'none'}",
        ] + list(board["notes"])
        render_report(
            f"Macro regime board ({region})",
            [
                Section(
                    "Indicators",
                    kv={"quadrant": board["quadrant"], "signals": str(board["risk_signals"])},
                    table=table,
                )
            ],
            caveats,
            subtitle=f"as of {board['as_of']}",
            path=html,
        )
        console.print(f"wrote {html}")


@macro_app.command("verify")
def verify(
    region: RegionOpt = "both",
    vendor: Annotated[
        str | None, typer.Option("--vendor", help="Only this vendor (fred | boe | ons).")
    ] = None,
    config_file: ConfigOption = None,
) -> None:
    """Fetch every catalogued series live and report which ids resolve (network; nothing is cached)."""
    from quant.macro.catalog import MACRO_SERIES, series_for_region
    from quant.macro.verify import summarize, verify_catalog

    svc = _service(config_file)
    specs = list(MACRO_SERIES.values()) if region == "both" else series_for_region(region)
    if vendor:
        specs = [s for s in specs if s.vendor == vendor]
    table = verify_catalog(svc.adapters, series=specs)
    console.print(
        frame_table(table.drop(columns=["frequency"]), title="macro catalogue verification", max_rows=200)
    )
    console.print(summarize(table))
    if (table["status"] == "fail").any():
        raise typer.Exit(code=1)
