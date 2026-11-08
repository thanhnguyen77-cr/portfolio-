"""`quant` command-line interface. Sub-commands are added as their stages land."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quant import __version__
from quant.cli.data_cmd import data_app
from quant.cli.universe_cmd import screen, universe_app

app = typer.Typer(
    no_args_is_help=True,
    help="Multi-asset quant research, backtesting and PAPER trading (US + UK). Research use only.",
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def config(
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="User YAML layered over the defaults.")
    ] = None,
) -> None:
    """Show the effective configuration (secrets masked)."""
    from quant.config.settings import load_settings

    settings = load_settings(config_file)
    typer.echo(settings.model_dump_json(indent=2))


app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")
app.command("screen")(screen)


if __name__ == "__main__":  # pragma: no cover
    app()
