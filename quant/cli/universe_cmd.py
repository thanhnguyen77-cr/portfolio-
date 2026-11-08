"""`quant universe ...` and `quant screen`: point-in-time universes and evidence-based screening."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

universe_app = typer.Typer(no_args_is_help=True, help="Build, inspect and import point-in-time universes.")


def _service(config_file):  # noqa: ANN001
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


@universe_app.command("build")
def build(
    name: Annotated[str, typer.Argument(help="sp500 | ndx100 | ftse100 | ftse250")],
    no_reconstruct: Annotated[
        bool, typer.Option("--no-reconstruct", help="Skip S&P change-table history.")
    ] = False,
    config_file: ConfigOption = None,
) -> None:
    """Fetch current constituents (Wikipedia), register instruments and record a membership snapshot."""
    from quant.core.errors import QuantError

    svc = _service(config_file)
    try:
        uni, info = svc.build_universe(name, reconstruct=not no_reconstruct)
    except (QuantError, RuntimeError) as exc:
        console.print(f"[red]could not build {name}: {exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(
        f"{name}: {info['members']} members as of {uni.snapshots()['as_of'].max()}; "
        f"+{len(info['added'])} / -{len(info['removed'])} vs previous snapshot; "
        f"quality [bold]{info['quality']}[/bold]; reconstructed intervals {info['reconstructed_intervals']}"
    )
    for note in info["notes"]:
        console.print(f"[yellow]{note}[/yellow]")


@universe_app.command("show")
def show(
    name: Annotated[str, typer.Argument()],
    as_of: Annotated[
        str | None, typer.Option("--as-of", help="Membership on this date (default today).")
    ] = None,
    config_file: ConfigOption = None,
) -> None:
    """List members at a date, the universe's quality flag and recorded snapshots."""
    svc = _service(config_file)
    uni = svc.universe(name)
    day = date.fromisoformat(as_of) if as_of else date.today()
    members = uni.membership_at(day)
    console.print(
        f"{name} @ {day}: {len(members)} members; quality [bold]{uni.quality()}[/bold]; "
        f"history from {uni.history_start()}"
    )
    if members:
        rows = [
            {"instrument_id": m, "symbol": (svc.catalog.get(m).symbol if svc.catalog.get(m) else "?")}
            for m in members
        ]
        console.print(frame_table(pd.DataFrame(rows), max_rows=600))
    snaps = uni.snapshots()
    if not snaps.empty:
        console.print(frame_table(snaps, title="snapshots"))


@universe_app.command("import")
def import_csv(
    name: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Argument(help="CSV with instrument_id, effective_from[, effective_to].")],
    complete: Annotated[bool, typer.Option("--complete", help="Mark the history as pit_complete.")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Import membership intervals from licensed or hand-checked data."""
    svc = _service(config_file)
    df = pd.read_csv(path)
    n = svc.universe(name).import_intervals(df, source=f"csv:{path.name}", complete=complete)
    console.print(f"{name}: imported {n} interval(s); quality [bold]{svc.universe(name).quality()}[/bold]")


def screen(
    refs: Annotated[
        list[str], typer.Argument(help="A universe name (sp500, ftse100, ...) or instrument refs.")
    ],
    start: Annotated[str, typer.Option("--start", "-s")] = "2023-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    min_adv: Annotated[
        float | None, typer.Option("--min-adv", help="Minimum median daily traded value (base ccy).")
    ] = None,
    no_refresh: Annotated[bool, typer.Option("--no-refresh", help="Use cached bars only.")] = False,
    html: Annotated[Path | None, typer.Option("--html", help="Also write an HTML report here.")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Rank instruments by liquidity, volatility band, price and data quality. Not a return forecast."""
    from quant.reports import Section, render_report
    from quant.universe.constituents import UNIVERSES
    from quant.universe.screen import ScreenConfig

    svc = _service(config_file)
    end_d = end or date.today().isoformat()
    members = svc.universe(refs[0]).membership_at(end_d) if len(refs) == 1 else []
    if members:
        refs = members
    elif len(refs) == 1 and refs[0] in UNIVERSES:
        console.print(
            f"[yellow]universe {refs[0]} is empty; run `quant universe build {refs[0]}` first[/yellow]"
        )
        raise typer.Exit(code=1)
    cfg = ScreenConfig(**({"min_adv_base": min_adv} if min_adv else {}))
    table = svc.screen(refs, start, end_d, config=cfg, refresh=False if no_refresh else None)
    if table.empty:
        console.print("[yellow]no bars available for the screen[/yellow]")
        raise typer.Exit(code=1)
    cols = [
        "instrument_id",
        "symbol",
        "score",
        "passes",
        "adv_base",
        "vol_ann",
        "price_base",
        "dq_grade",
        "reasons",
    ]
    console.print(
        frame_table(
            table[cols], title=f"screen ({int(table['passes'].sum())}/{len(table)} pass)", max_rows=600
        )
    )
    if html:
        caveats = [
            "The screen measures tradability and data quality; it does not forecast returns.",
            f"Thresholds: {cfg.model_dump()}",
            "ADV is the median of close x volume over the window, converted at vendor FX closes.",
        ]
        render_report(
            "Universe screen",
            [Section("Ranked candidates", table=table)],
            caveats,
            path=html,
            subtitle=f"{len(table)} instruments, {start}..{end_d}",
        )
        console.print(f"wrote {html}")
