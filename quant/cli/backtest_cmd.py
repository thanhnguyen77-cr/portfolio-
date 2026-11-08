"""`quant backtest run`: backtest a reference strategy from the cache, with optional robustness analysis."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer

from quant.cli._common import ConfigOption, console, frame_table, settings_from

backtest_app = typer.Typer(
    no_args_is_help=True, help="Run backtests and robustness analysis from cached data."
)

STRATEGIES: dict[str, str] = {
    "buy_and_hold": "quant.strategies:BuyAndHold",
    "fixed_weights": "quant.strategies:FixedWeights",
    "trend_following": "quant.strategies:TrendFollowing",
    "xs_momentum": "quant.strategies:CrossSectionalMomentum",
    "pairs_cointegration": "quant.strategies:PairsCointegration",
}


def _load_strategy_class(name: str):  # noqa: ANN202
    import importlib

    if name not in STRATEGIES:
        raise typer.BadParameter(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    module, cls = STRATEGIES[name].split(":")
    return getattr(importlib.import_module(module), cls)


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_params(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise typer.BadParameter(f"--param expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = _coerce(v.strip())
    return out


def parse_grid(spec: str | None) -> dict[str, list[Any]]:
    """'fast=10,20,50;slow=100,200' -> {'fast': [10, 20, 50], 'slow': [100, 200]}"""
    grid: dict[str, list[Any]] = {}
    for part in filter(None, (spec or "").split(";")):
        if "=" not in part:
            raise typer.BadParameter(f"--sensitivity expects key=v1,v2;key2=...; got {part!r}")
        k, vals = part.split("=", 1)
        grid[k.strip()] = [_coerce(v.strip()) for v in vals.split(",") if v.strip()]
    return grid


@backtest_app.command("run")
def run(
    strategy: Annotated[str, typer.Argument(help="buy_and_hold | trend_following | xs_momentum")],
    instruments: Annotated[
        str | None, typer.Option("--instruments", "-i", help="Comma-separated refs.")
    ] = None,
    universe: Annotated[
        str | None, typer.Option("--universe", "-u", help="Use a stored universe's members.")
    ] = None,
    start: Annotated[str, typer.Option("--start", "-s")] = "2020-01-01",
    end: Annotated[str | None, typer.Option("--end", "-e")] = None,
    cash: Annotated[float, typer.Option("--cash", help="Initial cash in the base currency.")] = 1_000_000.0,
    base: Annotated[str | None, typer.Option("--base", help="Base currency; default from config.")] = None,
    params: Annotated[
        list[str] | None, typer.Option("--param", "-p", help="Strategy parameter key=value.")
    ] = None,
    allow_short: Annotated[bool, typer.Option("--allow-short")] = False,
    no_refresh: Annotated[bool, typer.Option("--no-refresh", help="Use cached bars only.")] = False,
    sensitivity: Annotated[
        str | None, typer.Option("--sensitivity", help="Grid: 'fast=10,20;slow=100,200'.")
    ] = None,
    walk_forward: Annotated[
        str | None, typer.Option("--walk-forward", help="'TRAIN,TEST' sessions, e.g. 252,63.")
    ] = None,
    bootstrap: Annotated[bool, typer.Option("--bootstrap", help="Block-bootstrap Sharpe interval.")] = False,
    html: Annotated[Path | None, typer.Option("--html", help="Write a tearsheet here.")] = None,
    inline_js: Annotated[
        bool, typer.Option("--inline-js", help="Embed plotly.js for a fully offline report.")
    ] = False,
    out_dir: Annotated[
        Path | None, typer.Option("--out", help="Persist equity/fills/orders parquet + json here.")
    ] = None,
    config_file: ConfigOption = None,
) -> None:
    """Backtest from the local cache. Prints the summary and caveats; never claims profitability."""
    from quant.backtest.analysis import (
        block_bootstrap_sharpe,
        deflated_sharpe_ratio,
        sensitivity_grid,
        sensitivity_summary,
    )
    from quant.backtest.analysis import (
        walk_forward as run_walk_forward,
    )
    from quant.backtest.engine import Backtest, BacktestConfig, trial_sharpes, variant_count
    from quant.backtest.runner import assemble, make_runner
    from quant.data.service import DataService
    from quant.reports.tearsheet import build_tearsheet

    settings = settings_from(config_file)
    svc = DataService(settings)
    end_d = date.fromisoformat(end) if end else date.today()
    start_d = date.fromisoformat(start)
    if universe:
        refs = svc.universe(universe).membership_at(end_d)
        if not refs:
            console.print(
                f"[yellow]universe {universe} is empty; run `quant universe build {universe}` first[/yellow]"
            )
            raise typer.Exit(code=1)
    elif instruments:
        refs = [r.strip() for r in instruments.split(",") if r.strip()]
    else:
        console.print("[red]give --instruments or --universe[/red]")
        raise typer.Exit(code=2)
    ids = [svc.resolve(r).instrument_id for r in refs]
    base_ccy = base or settings.trading.base_currency.value
    strategy_params = parse_params(params)
    cls = _load_strategy_class(strategy)
    config = BacktestConfig(
        start=start_d, end=end_d, initial_cash=cash, base_currency=base_ccy, allow_short=allow_short
    )
    parts = assemble(
        svc,
        ids,
        start_d,
        end_d,
        base_currency=base_ccy,
        refresh=False if no_refresh else None,
        universe=universe,
    )
    usable = [i for i in ids if i not in parts["missing"]]
    if not usable:
        console.print("[red]no price data for the requested instruments[/red]")
        raise typer.Exit(code=1)

    def factory(p):  # noqa: ANN001, ANN202
        return cls(usable, **p)

    result = Backtest(
        config,
        factory(strategy_params),
        parts["bars"],
        parts["instruments"],
        actions=parts["actions"],
        fx=parts["fx"],
        caveats=parts["caveats"],
    ).run()
    console.print(result.summary())

    robustness: dict[str, Any] = {}
    runner = make_runner(
        factory, config, parts["bars"], parts["instruments"], actions=parts["actions"], fx=parts["fx"]
    )
    if sensitivity:
        grid = parse_grid(sensitivity)
        table = sensitivity_grid(runner, strategy_params, grid, start_d, end_d)
        robustness["sensitivity"] = table
        robustness["sensitivity_summary"] = sensitivity_summary(table)
        console.print(frame_table(table, title="parameter sensitivity"))
    if walk_forward:
        train, test = (int(x) for x in walk_forward.split(","))
        grid_points = [strategy_params]
        if sensitivity:
            import itertools

            g = parse_grid(sensitivity)
            grid_points = [
                {**strategy_params, **dict(zip(g, combo, strict=True))}
                for combo in itertools.product(*g.values())
            ]
        sessions = result.equity_series.index
        wf = run_walk_forward(runner, grid_points, sessions, train, test)
        robustness["walk_forward"] = wf
        console.print(
            frame_table(wf.windows.assign(params=wf.windows["params"].map(str)), title="walk-forward")
        )
        console.print(f"OOS: {wf.oos_metrics}")
    rets = result.equity_series.pct_change().dropna()
    if bootstrap and len(rets) >= 40:
        robustness["bootstrap"] = block_bootstrap_sharpe(rets)
        console.print(f"bootstrap Sharpe: {robustness['bootstrap']}")
    if len(rets) >= 10:
        robustness["deflated_sharpe"] = deflated_sharpe_ratio(
            rets, variant_count(result.strategy["name"]), trial_sharpes=trial_sharpes(result.strategy["name"])
        )
        d = robustness["deflated_sharpe"]
        console.print(
            f"deflated Sharpe: {d['deflated_sharpe']:.1%} after {d['n_trials']} variant(s) "
            f"(expected max noise Sharpe {d['expected_max_sharpe_annual']:.2f})"
        )
    if html:
        build_tearsheet(result, robustness=robustness, path=html, inline_plotly=inline_js)
        console.print(f"wrote {html}")
    if out_dir:
        result.to_dir(out_dir)
        console.print(f"wrote {out_dir}")
