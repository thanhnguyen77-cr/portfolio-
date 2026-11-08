"""FactorTilt and MacroRegimeSwitch: planted signals, parameter validation, and a run through the engine."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, zero_costs

from quant.backtest.engine import Backtest, BacktestConfig
from quant.backtest.strategy import Context
from quant.strategies import FactorTilt, MacroRegimeSwitch

S, E = date(2023, 1, 3), date(2024, 6, 28)


def _ctx_from_bars(catalog, bars: pd.DataFrame, t: date, ids: list[str]) -> Context:
    by_id = {k: g.sort_values("session_date") for k, g in bars.groupby("instrument_id")}

    def history(iid, n):  # noqa: ANN001,ANN202
        g = by_id[iid]
        g = g[g["session_date"] <= pd.Timestamp(t)].copy()
        g["adj_close"] = g["close"]
        return g.tail(n) if n else g

    return Context(
        session_date=t,
        instruments={i: catalog[i] for i in ids},
        base_currency="USD",
        positions={},
        cash={"USD": 1e6},
        equity=1e6,
        allow_short=False,
        fractional_shares=True,
        min_notional=0.0,
        _history=history,
        _close=lambda iid: float(by_id[iid]["close"].iloc[-1]),
        _fx=lambda a, b: 1.0,
    )


def test_factor_tilt_picks_the_planted_winner(catalog):
    ids = ["XNAS_AAPL", "XNAS_MSFT", "XNYS_JPM", "XNYS_XOM"]
    frames = []
    for k, iid in enumerate(ids):
        # AAPL: strong drift, low vol -> top on both momentum and low_vol; XOM: negative drift, high vol
        drift = {0: 0.004, 1: 0.0003, 2: 0.0001, 3: -0.006}[k]  # ~ +0.44 / -0.66 log-moves over the window
        vol = {0: 0.004, 1: 0.010, 2: 0.010, 3: 0.020}[k]  # so the planted ranks dominate the noise
        frames.append(make_bars(catalog[iid], S, E, seed=k + 1, drift=drift, vol=vol))
    bars = pd.concat(frames)
    strat = FactorTilt(ids, top_n=1, lookback=120, skip=10, vol_window=120)
    strat.on_start({i: catalog[i] for i in ids})
    ctx = _ctx_from_bars(catalog, bars, date(2024, 6, 3), ids)
    scores = strat.scores(ctx)
    assert scores.idxmax() == "XNAS_AAPL" and scores.idxmin() == "XNYS_XOM"
    weights = strat.on_session(ctx)
    assert weights == {"XNAS_AAPL": 1.0, "XNAS_MSFT": 0.0, "XNYS_JPM": 0.0, "XNYS_XOM": 0.0}
    assert strat.on_session(ctx) is None  # same month: no second rebalance
    with pytest.raises(ValueError, match="price-based"):
        FactorTilt(ids, factors=("value",))
    with pytest.raises(ValueError, match="lookback"):
        FactorTilt(ids, lookback=10, skip=21)
    sector = FactorTilt(
        ids, top_n=2, lookback=120, skip=10, vol_window=120, sector_neutral=True, rebalance="weekly"
    )
    sector.on_start({i: catalog[i] for i in ids})
    w = sector.on_session(ctx)
    assert (
        w is not None and sum(w.values()) == pytest.approx(1.0) and len([v for v in w.values() if v > 0]) == 2
    )
    cfg = BacktestConfig(start=S, end=E, cost_model=zero_costs(), fractional_shares=True, min_notional=0.0)
    res = Backtest(cfg, FactorTilt(ids, top_n=2, lookback=120, skip=10, vol_window=120), bars, catalog).run()
    assert len(res.fills) > 0 and res.equity["equity"].notna().all()
    assert res.strategy["params"]["factors"] == ["momentum", "low_vol"]


def test_macro_regime_switch_follows_the_benchmark(catalog):
    ids = ["XNAS_AAPL", "XNAS_MSFT", "XNYS_SPX", "XNYS_VIX"]
    up_spx = make_bars(catalog["XNYS_SPX"], S, E, seed=1, drift=0.0015, vol=0.004, start_price=4000.0)
    calm_vix = make_bars(catalog["XNYS_VIX"], S, E, seed=2, drift=0.0, vol=0.0, start_price=14.0)
    stocks = pd.concat([make_bars(catalog[i], S, E, seed=k + 3) for k, i in enumerate(ids[:2])])
    bars_on = pd.concat([stocks, up_spx, calm_vix])
    strat = MacroRegimeSwitch(
        ids, benchmark="XNYS_SPX", vol_index="XNYS_VIX", risk_on={"XNAS_AAPL": 0.5, "XNAS_MSFT": 0.4}
    )
    t = date(2024, 6, 3)
    assert strat.regime(_ctx_from_bars(catalog, bars_on, t, ids)) == "risk_on"
    w = strat.on_session(_ctx_from_bars(catalog, bars_on, t, ids))
    assert w == {"XNAS_AAPL": 0.5, "XNAS_MSFT": 0.4, "XNYS_SPX": 0.0, "XNYS_VIX": 0.0}
    assert strat.on_session(_ctx_from_bars(catalog, bars_on, t, ids)) is None  # same label, same week
    down_spx = make_bars(catalog["XNYS_SPX"], S, E, seed=1, drift=-0.002, vol=0.004, start_price=4000.0)
    wild_vix = make_bars(catalog["XNYS_VIX"], S, E, seed=2, drift=0.0, vol=0.0, start_price=40.0)
    bars_off = pd.concat([stocks, down_spx, wild_vix])
    assert strat.regime(_ctx_from_bars(catalog, bars_off, t, ids)) == "risk_off"
    off = strat.on_session(_ctx_from_bars(catalog, bars_off, t, ids))
    assert off == {"XNAS_AAPL": 0.0, "XNAS_MSFT": 0.0, "XNYS_SPX": 0.0, "XNYS_VIX": 0.0}
    caution = strat.weights_for("caution")
    assert caution["XNAS_AAPL"] == pytest.approx(0.25) and caution["XNAS_MSFT"] == pytest.approx(0.2)
    hedged = MacroRegimeSwitch(
        ids, benchmark="XNYS_SPX", risk_on={"XNAS_AAPL": 0.8}, risk_off={"XNAS_MSFT": 0.5}, caution_scale=0.5
    )
    assert hedged.weights_for("caution") == {
        "XNAS_AAPL": 0.4,
        "XNAS_MSFT": 0.25,
        "XNYS_SPX": 0.0,
        "XNYS_VIX": 0.0,
    }
    with pytest.raises(ValueError, match="benchmark"):
        MacroRegimeSwitch(ids[:2], benchmark="XNYS_SPX", risk_on={"XNAS_AAPL": 1.0})
    with pytest.raises(ValueError, match="outside the list"):
        MacroRegimeSwitch(ids, benchmark="XNYS_SPX", risk_on={"XNYS_JPM": 1.0})
    cfg = BacktestConfig(start=S, end=E, cost_model=zero_costs(), fractional_shares=True, min_notional=0.0)
    res = Backtest(cfg, strat, bars_on, catalog).run()
    assert res.equity["equity"].notna().all() and len(res.fills) >= 2
    assert set(res.fills["instrument_id"]) <= {"XNAS_AAPL", "XNAS_MSFT"}  # indices are never traded
    assert np.isfinite(res.metrics["sharpe"])
