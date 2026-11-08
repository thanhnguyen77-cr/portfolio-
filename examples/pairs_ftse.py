"""Cointegration tests on two FTSE names and a pairs backtest.

Run:  python examples/pairs_ftse.py
"""

from __future__ import annotations

from datetime import date

from quant.backtest.engine import BacktestConfig
from quant.backtest.runner import run_from_service
from quant.config.settings import load_settings
from quant.data.service import DataService
from quant.stats.cointegration import engle_granger, half_life, johansen
from quant.strategies import PairsCointegration

A, B = "SHEL.L", "BP.L"
START, END = date(2016, 1, 4), date(2024, 6, 28)


def main() -> None:
    svc = DataService(load_settings())
    ids = [svc.resolve(A).instrument_id, svc.resolve(B).instrument_id]
    panel = svc.close_panel(ids, START, END, to_currency="GBP").dropna()
    eg = engle_granger(panel[ids[0]], panel[ids[1]])
    print(f"Engle-Granger: stat {eg.statistic:.2f}, p {eg.pvalue:.3f}, hedge ratio {eg.hedge_ratio:.3f}")
    print(f"half-life of the spread: {half_life(eg.spread):.1f} sessions")
    jo = johansen(panel[ids])
    print(f"Johansen trace stats: {jo.trace_stats}")
    cfg = BacktestConfig(start=date(2018, 1, 2), end=END, base_currency="GBP", allow_short=True)
    result = run_from_service(svc, PairsCointegration(ids), cfg, refresh=False)
    print(result.summary())


if __name__ == "__main__":
    main()
