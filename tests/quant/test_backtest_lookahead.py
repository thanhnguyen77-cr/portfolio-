"""Look-ahead guards: poisoning the future must not change past decisions; running day by day on truncated
data must reproduce the full-sample run."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import actions_frame, make_bars, zero_costs

from quant.backtest.engine import Backtest, BacktestConfig
from quant.data.fx import FXRates
from quant.strategies import TrendFollowing

S, E = date(2023, 6, 1), date(2024, 6, 28)
IDS = ["XNAS_AAPL", "XNYS_JPM", "XLON_VOD"]


@pytest.fixture
def world(catalog):
    bars = pd.concat(
        [
            make_bars(catalog["XNAS_AAPL"], S, E, seed=31, vol=0.015),
            make_bars(catalog["XNYS_JPM"], S, E, seed=32, vol=0.012),
            make_bars(catalog["XLON_VOD"], S, E, seed=33, start_price=0.8, vol=0.015),
        ]
    )
    fxbars = make_bars(catalog["FX_GBPUSD"], S, E, seed=34, start_price=1.26, vol=0.004)
    ex1 = bars[bars["instrument_id"] == "XNAS_AAPL"]["session_date"].iloc[60]
    ex2 = bars[bars["instrument_id"] == "XLON_VOD"]["session_date"].iloc[200]
    acts = pd.concat(
        [
            actions_frame("XNAS_AAPL", [(ex1, "dividend", 0.25, "USD")]),
            actions_frame("XLON_VOD", [(ex2, "dividend", 0.02, "GBP")]),
        ],
        ignore_index=True,
    )
    return bars, fxbars, acts


def _run(catalog, bars, fxbars, acts, start=S, end=E):
    cfg = BacktestConfig(
        start=start, end=end, initial_cash=250_000.0, cost_model=zero_costs(), fractional_shares=False
    )
    strat = TrendFollowing(IDS, fast=5, slow=20)
    return Backtest(cfg, strat, bars, catalog, actions=acts, fx=FXRates.from_bars(fxbars, catalog)).run()


def _decisions(res, upto: pd.Timestamp, strict: bool = False):
    o = res.orders.copy()
    o["created_session"] = pd.to_datetime(o["created_session"])
    o = o[o["created_session"] < upto] if strict else o[o["created_session"] <= upto]
    return o[["created_session", "instrument_id", "side", "qty"]].reset_index(drop=True)


def test_poisoned_future_leaves_past_decisions_unchanged(catalog, world):
    bars, fxbars, acts = world
    full = _run(catalog, bars, fxbars, acts)
    assert len(full.orders) >= 6, "fixture must trade enough to be meaningful"
    sessions = full.equity["session_date"]
    cut = sessions.iloc[len(sessions) // 2]
    rng = np.random.default_rng(99)
    poisoned = bars.copy()
    future = poisoned["session_date"] > cut
    scale = rng.uniform(0.2, 5.0, future.sum())
    for col in ("open", "high", "low", "close"):
        poisoned.loc[future, col] = poisoned.loc[future, col].to_numpy() * scale
    poisoned.loc[future, "volume"] = poisoned.loc[future, "volume"].to_numpy() * rng.uniform(
        0.1, 10, future.sum()
    )
    fx_poisoned = fxbars.copy()
    fx_future = fx_poisoned["session_date"] > cut
    fx_poisoned.loc[fx_future, "close"] = fx_poisoned.loc[fx_future, "close"].to_numpy() * rng.uniform(
        0.5, 2, fx_future.sum()
    )
    acts_poisoned = acts[pd.to_datetime(acts["ex_date"]) <= cut]
    res_p = _run(catalog, poisoned, fx_poisoned, acts_poisoned)
    pd.testing.assert_frame_equal(_decisions(full, cut), _decisions(res_p, cut))
    eq_full = full.equity[full.equity["session_date"] <= cut]["equity"].to_numpy()
    eq_p = res_p.equity[res_p.equity["session_date"] <= cut]["equity"].to_numpy()
    np.testing.assert_allclose(eq_full, eq_p, rtol=1e-12)
    assert not np.allclose(full.equity["equity"].iloc[-1], res_p.equity["equity"].iloc[-1])


def test_incremental_runs_reproduce_the_batch_run(catalog, world):
    bars, fxbars, acts = world
    full = _run(catalog, bars, fxbars, acts)
    sessions = full.equity["session_date"]
    for frac in (0.3, 0.6, 0.9):
        cut = sessions.iloc[int(len(sessions) * frac)]
        trunc = _run(
            catalog,
            bars[bars["session_date"] <= cut],
            fxbars[fxbars["session_date"] <= cut],
            acts[pd.to_datetime(acts["ex_date"]) <= cut],
            end=cut.date(),
        )
        eq_full = full.equity[full.equity["session_date"] <= cut]["equity"].to_numpy()
        np.testing.assert_allclose(eq_full, trunc.equity["equity"].to_numpy(), rtol=1e-12)
        pd.testing.assert_frame_equal(_decisions(full, cut, strict=True), _decisions(trunc, cut, strict=True))
