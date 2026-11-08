"""`quant demo`: the whole pipeline, offline, on a synthetic mini-dataset."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from quant.cli._common import ConfigOption, console, settings_from


def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to write the reports.")] = Path(
        "data/reports/demo"
    ),
    seed: Annotated[int, typer.Option("--seed")] = 42,
    start: Annotated[str, typer.Option("--start")] = "2020-01-02",
    end: Annotated[str, typer.Option("--end")] = "2024-06-28",
    no_backtests: Annotated[
        bool, typer.Option("--no-backtests", help="Skip the three backtests (faster).")
    ] = False,
    config_file: ConfigOption = None,
) -> None:
    """Generate SYNTHETIC data and run every stage end to end; exits 1 if any step failed."""
    from quant.demo import Demo

    settings = settings_from(config_file)
    runner = Demo(
        settings,
        out,
        seed=seed,
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        backtests=not no_backtests,
    )
    console.print(
        f"[yellow]SYNTHETIC DATA demo (seed {seed}); nothing here describes a real market.[/yellow]"
    )
    result = runner.run()
    console.print(result.summary())
    console.print(
        "explore the data with the demo vendor, e.g. "
        "QUANT_DATA__ADAPTER_PRIORITY='[\"demo\"]' quant data show XNAS_US01 --tail 5"
    )
    if not result.ok:
        raise typer.Exit(code=1)
