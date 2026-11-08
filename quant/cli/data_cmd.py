"""`quant data ...`: fetch, inspect and audit cached market data."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import pandas as pd
import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

data_app = typer.Typer(no_args_is_help=True, help="Fetch, inspect and audit cached market data.")


def _service(config_file):  # noqa: ANN001 - typer passes Path | None
    from quant.data.service import DataService

    return DataService(settings_from(config_file))


@data_app.command("fetch")
def fetch(
    refs: Annotated[list[str], typer.Argument(help="Instrument ids, canonical symbols or vendor symbols.")],
    start: Annotated[str, typer.Option("--start", "-s", help="First session (YYYY-MM-DD).")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e", help="Last session; default today.")] = None,
    vendor: Annotated[str | None, typer.Option("--vendor", help="Adapter name; default from config.")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Download daily bars into the local cache (incremental; records lineage)."""
    svc = _service(config_file)
    end_d = date.fromisoformat(end) if end else date.today()
    report = svc.fetch_bars(refs, date.fromisoformat(start), end_d, vendor=vendor)
    console.print(report.summary())
    if report.writes:
        rows = pd.DataFrame([w.as_dict() for w in report.writes]).drop(columns=["path"])
        console.print(frame_table(rows, title="writes"))
    for iid, err in sorted(report.errors.items()):
        console.print(f"[red]{iid}[/red]: {err}")
    if report.errors and not report.writes:
        raise typer.Exit(code=1)


@data_app.command("show")
def show(
    ref: Annotated[str, typer.Argument(help="Instrument id or symbol.")],
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    tail: Annotated[int, typer.Option("--tail", "-n", help="Rows to print from the end.")] = 10,
    vendor: Annotated[str | None, typer.Option("--vendor")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Print cached bars for one instrument (fetching what is missing first)."""
    svc = _service(config_file)
    bars = svc.get_bars([ref], start, end, vendor=vendor)
    if bars.empty:
        console.print(f"[yellow]no bars for {ref}[/yellow]")
        raise typer.Exit(code=1)
    cols = ["session_date", "open", "high", "low", "close", "volume", "currency", "adjustment_state"]
    title = f"{bars['instrument_id'].iloc[0]} ({len(bars)} rows)"
    console.print(frame_table(bars[cols].tail(tail), title=title))


@data_app.command("instruments")
def instruments(
    asset_class: Annotated[str | None, typer.Option("--asset-class", "-a")] = None,
    market: Annotated[str | None, typer.Option("--market", "-m")] = None,
    config_file: ConfigOption = None,
) -> None:
    """List the instrument catalogue."""
    from quant.core.catalog import InstrumentCatalog

    cat = InstrumentCatalog.from_yaml(settings_from(config_file).instruments_file)
    rows = [
        {
            "instrument_id": i.instrument_id,
            "symbol": i.symbol,
            "asset_class": i.asset_class.value,
            "market": i.market.value,
            "currency": i.currency.value,
            "scale": i.price_scale,
            "tradable": i.tradable,
            "yahoo": i.identifiers.get("yahoo", ""),
        }
        for i in cat.filter(asset_class=asset_class, market=market)
    ]
    console.print(frame_table(pd.DataFrame(rows), title=f"{len(rows)} instrument(s)", max_rows=200))


@data_app.command("cached")
def cached(config_file: ConfigOption = None) -> None:
    """Show which series are cached, their ranges and when they were last checked."""
    svc = _service(config_file)
    df = svc.cached_series()
    if df.empty:
        console.print("[yellow]cache is empty[/yellow]")
        return
    console.print(frame_table(df, title="cached series", max_rows=200))


@data_app.command("validate")
def validate(
    refs: Annotated[list[str], typer.Argument(help="Instrument ids or symbols.")],
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Fetch missing data before validating.")] = False,
    issues: Annotated[bool, typer.Option("--issues", help="Also list individual issues.")] = False,
    strict: Annotated[bool, typer.Option("--strict", help="Exit 1 if any instrument grades F.")] = False,
    config_file: ConfigOption = None,
) -> None:
    """Data-quality report for cached bars: gaps, stale closes, outliers, jumps, dividends, volume."""
    svc = _service(config_file)
    report = svc.validate(refs, start, end, refresh=True if refresh else False)
    table = report.to_frame()
    if table.empty:
        console.print("[yellow]no cached bars to validate (use --refresh to fetch)[/yellow]")
        raise typer.Exit(code=1)
    cols = [
        "instrument_id",
        "grade",
        "score",
        "bars",
        "gaps",
        "stale_days",
        "outliers",
        "ohlc_violations",
        "denomination_changes",
        "unexplained_jumps",
        "dividend_warnings",
        "issues",
    ]
    console.print(frame_table(table[cols], title="data quality"))
    if issues and not report.issues_frame().empty:
        console.print(frame_table(report.issues_frame(), title="issues", max_rows=200))
    if strict and not report.ok("C"):
        raise typer.Exit(code=1)


@data_app.command("macro")
def macro(
    series_id: Annotated[str, typer.Argument(help="Series code, e.g. DGS10 (FRED) or IUDBEDR (BoE).")],
    start: Annotated[str, typer.Option("--start", "-s")] = "2015-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    vendor: Annotated[str | None, typer.Option("--vendor", help="fred | boe (inferred if omitted)")] = None,
    tail: Annotated[int, typer.Option("--tail", "-n")] = 10,
    config_file: ConfigOption = None,
) -> None:
    """Fetch (incrementally) and print a macro series."""
    svc = _service(config_file)
    series = svc.get_macro(series_id, start, end, vendor=vendor)
    if series.empty:
        console.print(f"[yellow]no observations for {series_id}[/yellow]")
        raise typer.Exit(code=1)
    df = series.tail(tail).rename_axis("date").reset_index()
    console.print(frame_table(df, title=f"{series_id} ({len(series)} observations)"))


@data_app.command("lineage")
def lineage(
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    config_file: ConfigOption = None,
) -> None:
    """Show the most recent data-lineage records."""
    svc = _service(config_file)
    df = svc.lineage.recent(limit)
    if df.empty:
        console.print("[yellow]no lineage yet[/yellow]")
        return
    cols = ["ts", "vendor", "operation", "rows", "instrument_ids", "sha256"]
    console.print(frame_table(df[cols], max_rows=limit))


@data_app.command("adapters")
def adapters(config_file: ConfigOption = None) -> None:
    """List data adapters: configured ones, keyed ones awaiting credentials, and licensed stubs."""
    svc = _service(config_file)
    console.print(frame_table(svc.adapter_inventory(), title="data adapters", max_rows=100))
    console.print(
        "[yellow]Licensed stubs raise AdapterNotConfigured with wiring steps; they are never selected "
        "automatically.[/yellow]"
    )


@data_app.command("filings")
def filings(
    refs: Annotated[list[str], typer.Argument(help="Instrument ids or symbols.")],
    vendor: Annotated[
        str | None, typer.Option("--vendor", help="sec_edgar | companies_house (auto).")
    ] = None,
    forms: Annotated[str | None, typer.Option("--forms", help="Comma-separated forms / categories.")] = None,
    since: Annotated[str | None, typer.Option("--since")] = None,
    config_file: ConfigOption = None,
) -> None:
    """Filings index with true filing dates (EDGAR submissions for US names, Companies House for UK)."""
    svc = _service(config_file)
    frame = svc.filings(
        refs, vendor=vendor, forms=[f.strip() for f in forms.split(",")] if forms else None, since=since
    )
    if frame.empty:
        console.print("[yellow]no filings returned[/yellow]")
        raise typer.Exit(code=1)
    show = frame[["instrument_id", "filed_at", "period_end", "form", "category", "description", "url"]]
    console.print(frame_table(show, title="filings", max_rows=200))
