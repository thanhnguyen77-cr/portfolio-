"""`quant` command-line interface. Sub-commands are added as their stages land."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from quant import __version__
from quant.cli.analyze_cmd import analyze_app
from quant.cli.backtest_cmd import backtest_app
from quant.cli.data_cmd import data_app
from quant.cli.factors_cmd import factors_app
from quant.cli.fundamentals_cmd import fundamentals_app
from quant.cli.macro_cmd import macro_app
from quant.cli.optimize_cmd import optimize_app
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
app.add_typer(backtest_app, name="backtest")
app.add_typer(analyze_app, name="analyze")
app.add_typer(fundamentals_app, name="fundamentals")
app.add_typer(factors_app, name="factors")
app.add_typer(macro_app, name="macro")
app.add_typer(optimize_app, name="optimize")
app.add_typer(universe_app, name="universe")
app.command("screen")(screen)


if __name__ == "__main__":  # pragma: no cover
    app()
