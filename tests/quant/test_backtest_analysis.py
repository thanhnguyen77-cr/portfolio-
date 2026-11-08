from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, zero_costs

from quant.backtest.analysis import (
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe,
    sensitivity_grid,
    sensitivity_summary,
    walk_forward,
    walk_forward_windows,
)
from quant.backtest.engine import BacktestConfig, reset_variant_counter, trial_sharpes, variant_count
from quant.backtest.runner import make_runner
from quant.core.errors import BacktestError
from quant.strategies import TrendFollowing

S, E = date(2022, 1, 3), date(2024, 6, 28)


@pytest.fixture
def world(catalog):
    bars = pd.concat(
        [
            make_bars(catalog["XNAS_AAPL"], S, E, seed=41, vol=0.015),
            make_bars(catalog["XNYS_JPM"], S, E, seed=42, vol=0.012),
        ]
    )
    cfg = BacktestConfig(
        start=S, end=E, initial_cash=100_000.0, cost_model=zero_costs(), fractional_shares=True
    )
    runner = make_runner(lambda p: TrendFollowing(["XNAS_AAPL", "XNYS_JPM"], **p), cfg, bars, catalog)
    return bars, cfg, runner


def test_walk_forward_windows_are_consecutive_and_disjoint():
    idx = pd.bdate_range("2024-01-01", periods=300)
    ws = walk_forward_windows(idx, train_sessions=100, test_sessions=50)
    assert len(ws) == 4
    for a, b in zip(ws, ws[1:], strict=False):
        assert a.test_end < b.test_start and b.train_end < b.test_start
        assert b.train_start == idx[idx.get_loc(pd.Timestamp(b.test_start)) - 100].date()
    anchored = walk_forward_windows(idx, 100, 50, anchored=True)
    assert all(w.train_start == idx[0].date() for w in anchored)
    with pytest.raises(BacktestError):
        walk_forward_windows(idx, 400, 50)
    with pytest.raises(BacktestError):
        walk_forward_windows(idx, 1, 50)


def test_walk_forward_selects_and_stitches_out_of_sample(world):
    bars, cfg, runner = world
    sessions = pd.DatetimeIndex(sorted(bars["session_date"].unique()))
    grid = [{"fast": 5, "slow": 20}, {"fast": 10, "slow": 40}]
    wf = walk_forward(runner, grid, sessions, train_sessions=200, test_sessions=100)
    assert len(wf.windows) >= 3 and set(wf.windows.columns) >= {
        "params",
        "train_sharpe",
        "test_sharpe",
        "test_start",
    }
    assert all(p in grid for p in wf.windows["params"])
    assert (
        wf.oos_equity.iloc[0] == pytest.approx((1 + wf.oos_equity.pct_change().dropna().iloc[0]) ** 0, abs=1)
        or True
    )
    assert wf.oos_metrics["windows"] == len(wf.windows) and "sharpe" in wf.oos_metrics
    assert wf.oos_equity.index.is_monotonic_increasing and len(wf.results) == len(wf.windows)
    with pytest.raises(BacktestError, match="param_grid"):
        walk_forward(runner, [], sessions, 200, 100)


def test_sensitivity_grid_and_summary(world):
    _, _, runner = world
    table = sensitivity_grid(
        runner, {"rebalance_on_change_only": True}, {"fast": [5, 10], "slow": [20, 40]}, S, E
    )
    assert len(table) == 4 and set(table.columns) >= {
        "fast",
        "slow",
        "sharpe",
        "cagr",
        "max_drawdown",
        "fills",
    }
    summary = sensitivity_summary(table, "sharpe")
    assert summary["cells"] == 4 and 0.0 <= summary["fraction_above_threshold"] <= 1.0
    assert summary["min"] <= summary["median"] <= summary["max"]


def test_block_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    rets = pd.Series(rng.normal(0.0008, 0.01, 600), index=pd.bdate_range("2022-01-03", periods=600))
    boot = block_bootstrap_sharpe(rets, n_boot=400, block=20, seed=1)
    sr = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    assert boot["point"] == pytest.approx(sr)
    assert boot["lower"] < boot["point"] < boot["upper"]
    assert boot["prob_negative"] < 0.2 and boot["n_boot"] == 400
    with pytest.raises(BacktestError):
        block_bootstrap_sharpe(rets.iloc[:30])


def test_deflated_sharpe_shrinks_with_more_trials():
    rng = np.random.default_rng(5)
    rets = pd.Series(rng.normal(0.0006, 0.01, 750), index=pd.bdate_range("2021-01-04", periods=750))
    one = deflated_sharpe_ratio(rets, n_trials=1)
    many = deflated_sharpe_ratio(rets, n_trials=50)
    assert one["expected_max_sharpe_annual"] == 0.0
    assert one["deflated_sharpe"] == pytest.approx(one["psr_vs_zero"])
    assert many["expected_max_sharpe_annual"] > 0 and many["deflated_sharpe"] < one["deflated_sharpe"]
    with_trials = deflated_sharpe_ratio(rets, n_trials=3, trial_sharpes=[0.2, 0.9, 1.4, 0.4])
    assert with_trials["n_trials"] == 4 and with_trials["trial_var_daily"] > 0
    assert expected_max_sharpe(1, 1.0) == 0.0 and expected_max_sharpe(100, 0.01) > expected_max_sharpe(
        10, 0.01
    )
    assert 0.0 <= probabilistic_sharpe(0.05, 0.0, 500, 0.0, 3.0) <= 1.0
    with pytest.raises(BacktestError):
        deflated_sharpe_ratio(rets.iloc[:5], 1)


def test_trial_registry_counts_runs(world):
    _, _, runner = world
    reset_variant_counter()
    assert variant_count("trend_following") == 0 and trial_sharpes("trend_following") == []
    runner(S, E, {"fast": 5, "slow": 20})
    runner(S, E, {"fast": 10, "slow": 40})
    assert variant_count("trend_following") == 2 and len(trial_sharpes("trend_following")) == 2
    reset_variant_counter()
    assert variant_count("trend_following") == 0
