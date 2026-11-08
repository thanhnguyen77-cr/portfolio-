"""Shared CLI plumbing: settings loading and rich output."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from quant.config.settings import QuantSettings, load_settings

console = Console()

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="User YAML layered over the package defaults.")
]


def settings_from(config_file: Path | None) -> QuantSettings:
    return load_settings(config_file)


def frame_table(df: pd.DataFrame, title: str | None = None, max_rows: int = 50) -> Table:
    table = Table(title=title, show_lines=False)
    for col in df.columns:
        table.add_column(str(col), overflow="fold")
    for _, row in df.head(max_rows).iterrows():
        table.add_row(*[_fmt(v) for v in row.tolist()])
    if len(df) > max_rows:
        table.caption = f"... {len(df) - max_rows} more row(s)"
    return table


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, pd.Timestamp):
        fmt = "%Y-%m-%d %H:%M%z" if (value.hour or value.minute) else "%Y-%m-%d"
        return value.strftime(fmt)
    return "" if value is None else str(value)
