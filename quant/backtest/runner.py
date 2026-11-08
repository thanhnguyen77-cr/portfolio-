"""Run backtests straight from the local data cache, with the caveats the data implies.

`run_from_service` assembles bars (with a warm-up buffer before `start`), corporate actions, FX pairs for
any non-base currency, a validation report and the universe's quality/coverage, then runs the engine and
attaches those facts to the result's Caveats.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd

from quant.backtest.engine import Backtest, BacktestConfig
from quant.backtest.result import BacktestResult
from quant.backtest.strategy import Strategy
from quant.core.time import to_date
from quant.data.fx import FXRates
from quant.data.service import DataService

WARMUP_CALENDAR_DAYS = 420  # ~ 1 trading year of history before `start` for signal warm-up


def assemble(
    service: DataService,
    instrument_ids: Sequence[str],
    start: date | str,
    end: date | str,
    *,
    base_currency: str,
    refresh: bool | None = None,
    universe: str | None = None,
) -> dict[str, Any]:
    """Bars, actions, FX, data-quality grades and universe facts for a run (no engine yet)."""
    start_d, end_d = to_date(start), to_date(end)
    fetch_start = start_d - timedelta(days=WARMUP_CALENDAR_DAYS)
    insts = [service.resolve(i) for i in instrument_ids]
    bars = service.get_bars(insts, fetch_start, end_d, refresh=refresh)
    ids = [i.instrument_id for i in insts]
    actions = service.db.load_corporate_actions(ids, vendor=None if bars.empty else bars["vendor"].iloc[0])
    foreign = sorted({i.currency.value for i in insts} - {base_currency})
    fx = None
    if foreign:
        needed = {frozenset({base_currency, f}) for f in foreign}
        pairs = [
            p
            for p in service.catalog.filter(asset_class="fx")
            if frozenset({p.symbol[:3], p.symbol[3:]}) in needed
        ]
        fx_bars = service.get_bars(pairs, fetch_start, end_d, refresh=refresh) if pairs else bars.iloc[0:0]
        fx = FXRates.from_bars(fx_bars, service.catalog)
    by_id = {i.instrument_id: i for i in insts}
    report = service.validate(insts, fetch_start, end_d, refresh=False) if not bars.empty else None
    caveats: list[str] = []
    if report is not None and report.results:
        grades = {iid: q.grade for iid, q in report.results.items()}
        worst = report.worst_grade()
        caveats.append(f"Data quality grades: {grades} (worst {worst}).")
    missing = [iid for iid in ids if bars.empty or iid not in set(bars["instrument_id"])]
    if missing:
        caveats.append(f"No price data for {missing}; they are excluded from the run.")
    if universe:
        uni = service.universe(universe)
        with_data = set() if bars.empty else set(bars["instrument_id"])
        cov = uni.coverage(start_d, end_d, with_data)
        caveats.append(
            f"Universe '{universe}' quality: {uni.quality()}; "
            f"coverage of historical members with price data: "
            f"{cov:.0%}. Results are an upper bound when coverage < 100% or quality is not pit_complete."
        )
    return {
        "instruments": {k: v for k, v in by_id.items() if k not in missing},
        "bars": bars,
        "actions": actions,
        "fx": fx,
        "report": report,
        "caveats": caveats,
        "missing": missing,
    }


def run_from_service(
    service: DataService,
    strategy: Strategy,
    config: BacktestConfig,
    *,
    refresh: bool | None = None,
    universe: str | None = None,
    benchmark: pd.Series | None = None,
) -> BacktestResult:
    parts = assemble(
        service,
        strategy.instruments,
        config.start,
        config.end,
        base_currency=config.base_currency.value,
        refresh=refresh,
        universe=universe,
    )
    if parts["missing"]:
        strategy.instruments = [i for i in strategy.instruments if i not in parts["missing"]]
    bt = Backtest(
        config,
        strategy,
        parts["bars"],
        parts["instruments"],
        actions=parts["actions"],
        fx=parts["fx"],
        benchmark=benchmark,
        caveats=parts["caveats"],
    )
    return bt.run()


def make_runner(
    strategy_factory: Callable[[Mapping[str, Any]], Strategy],
    config: BacktestConfig,
    bars: pd.DataFrame,
    instruments: Mapping[str, Any],
    *,
    actions: pd.DataFrame | None = None,
    fx: FXRates | None = None,
) -> Callable[[date, date, Mapping[str, Any]], BacktestResult]:
    """A `Runner` for walk-forward / sensitivity: (start, end, params) -> BacktestResult on fixed data."""

    def _run(start: date, end: date, params: Mapping[str, Any]) -> BacktestResult:
        cfg = config.model_copy(update={"start": to_date(start), "end": to_date(end)})
        return Backtest(cfg, strategy_factory(params), bars, instruments, actions=actions, fx=fx).run()

    return _run
