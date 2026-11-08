"""Cross-sectional momentum over a mixed US/UK list, GBP base, costs on, with robustness checks.

Run:  python examples/us_uk_momentum.py   (fetches daily bars through the configured adapters)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from quant.backtest.analysis import block_bootstrap_sharpe, deflated_sharpe_ratio, walk_forward
from quant.backtest.engine import BacktestConfig, trial_sharpes, variant_count
from quant.backtest.runner import assemble, make_runner, run_from_service
from quant.config.settings import load_settings
from quant.data.service import DataService
from quant.reports.tearsheet import build_tearsheet
from quant.strategies import CrossSectionalMomentum

REFS = ["AAPL", "MSFT", "JPM", "XOM", "VOD.L", "SHEL.L", "BP.L", "HSBA.L", "AZN.L", "ULVR.L"]
START, END = date(2018, 1, 2), date(2024, 6, 28)


def main() -> None:
    svc = DataService(load_settings())
    ids = [svc.resolve(r).instrument_id for r in REFS]
    cfg = BacktestConfig(start=START, end=END, base_currency="GBP", initial_cash=1_000_000.0)
    strategy = CrossSectionalMomentum(ids, top_fraction=0.4)
    result = run_from_service(svc, strategy, cfg)
    print(result.summary())

    parts = assemble(svc, ids, START, END, base_currency="GBP", refresh=False)
    runner = make_runner(
        lambda p: CrossSectionalMomentum(ids, **p), cfg, parts["bars"], parts["instruments"],
        actions=parts["actions"], fx=parts["fx"],
    )  # fmt: skip
    sessions = result.equity_series.index
    grid = [{"top_fraction": f} for f in (0.3, 0.4, 0.5)]
    wf = walk_forward(runner, grid, sessions, train_sessions=252, test_sessions=63)
    rets = result.equity_series.pct_change().dropna()
    robustness = {
        "walk_forward": wf,
        "bootstrap": block_bootstrap_sharpe(rets, n_boot=500, seed=0),
        "deflated_sharpe": deflated_sharpe_ratio(
            rets, variant_count(strategy.name), trial_sharpes=trial_sharpes(strategy.name)
        ),
    }
    out = Path("data/reports/us_uk_momentum.html")
    build_tearsheet(result, robustness=robustness, path=out)
    print(f"tearsheet: {out}")


if __name__ == "__main__":
    main()
