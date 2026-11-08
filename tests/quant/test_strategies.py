from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from quant.backtest.engine import Backtest, BacktestConfig
from quant.backtest.strategy import Context
from quant.core.errors import BacktestError
from quant.strategies import BuyAndHold, TrendFollowing
from quant.technical.trend import ema, sma


def _ctx(catalog, positions=None, equity=10_000.0, closes=None, allow_short=False, fractional=False):
    closes = closes or {"XNAS_AAPL": 100.0, "XNYS_JPM": 50.0}
    return Context(
        session_date=date(2024, 1, 2),
        instruments={k: catalog[k] for k in closes},
        base_currency="USD",
        positions=positions or {},
        cash={"USD": equity},
        equity=equity,
        allow_short=allow_short,
        fractional_shares=fractional,
        min_notional=100.0,
        _history=lambda iid, n: pd.DataFrame(),
        _close=lambda iid: closes.get(iid),
        _fx=lambda a, b: 1.0,
    )


def test_target_weights_sizing(catalog):
    ctx = _ctx(catalog)
    orders = ctx.target_weights({"XNAS_AAPL": 0.5, "XNYS_JPM": 0.25})
    sized = {o.instrument_id: (o.side.value, o.qty) for o in orders}
    assert sized == {"XNAS_AAPL": ("buy", 50.0), "XNYS_JPM": ("buy", 50.0)}
    ctx2 = _ctx(catalog, positions={"XNAS_AAPL": 60.0})
    [sell] = ctx2.target_weights({"XNAS_AAPL": 0.5})
    assert sell.side.value == "sell" and sell.qty == 10.0
    assert ctx2.target_weights({"XNAS_AAPL": 0.6005}) == []  # below min_notional
    assert ctx2.weight("XNAS_AAPL") == pytest.approx(0.6)
    with pytest.raises(BacktestError, match="allow_short"):
        ctx.target_weights({"XNAS_AAPL": -0.1})
    with pytest.raises(BacktestError, match="unknown instrument"):
        ctx.target_weights({"XNAS_NOPE": 0.1})
    [short] = _ctx(catalog, allow_short=True).target_weights({"XNAS_AAPL": -0.5})
    assert short.side.value == "sell" and short.qty == 50.0


def test_sma_ema_basics():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0] and np.isnan(sma(s, 3).iloc[1])
    e = ema(s, 3)
    assert np.isnan(e.iloc[1]) and e.iloc[-1] == pytest.approx(4.0625)
    with pytest.raises(ValueError):
        sma(s, 0)


def test_trend_following_goes_long_in_uptrend_and_flat_in_downtrend(catalog):
    up = make_bars(catalog["XNAS_AAPL"], date(2024, 1, 2), date(2024, 6, 28), vol=0.0, drift=0.002)
    strat = TrendFollowing(["XNAS_AAPL"], fast=5, slow=20)
    cfg = BacktestConfig(
        start=date(2024, 1, 2), end=date(2024, 6, 28), initial_cash=10_000, fractional_shares=True
    )
    res = Backtest(cfg, strat, up, catalog).run()
    assert len(res.fills) == 1 and res.fills["side"].iloc[0] == "buy"  # one entry, then no drift trading
    assert res.equity["gross_exposure"].iloc[-1] > 0.95
    down = make_bars(catalog["XNAS_AAPL"], date(2024, 1, 2), date(2024, 6, 28), vol=0.0, drift=-0.002)
    res_d = Backtest(cfg, TrendFollowing(["XNAS_AAPL"], fast=5, slow=20), down, catalog).run()
    assert res_d.fills.empty and res_d.equity["equity"].iloc[-1] == 10_000
    with pytest.raises(ValueError):
        TrendFollowing(["XNAS_AAPL"], fast=20, slow=5)


def test_buy_and_hold_waits_for_prices_and_trades_once(catalog):
    s, e = date(2024, 7, 1), date(2024, 7, 12)
    bars = pd.concat(
        [make_bars(catalog["XNAS_AAPL"], s, e), make_bars(catalog["XLON_VOD"], s, e, start_price=0.7)]
    )
    from quant.data.fx import FXRates

    fx = FXRates({"GBPUSD": pd.Series(1.27, index=pd.bdate_range(s, e))})
    strat = BuyAndHold(["XNAS_AAPL", "XLON_VOD"], weights={"XNAS_AAPL": 0.6, "XLON_VOD": 0.4})
    cfg = BacktestConfig(start=s, end=e, initial_cash=100_000, fractional_shares=True)
    res = Backtest(cfg, strat, bars, catalog, fx=fx).run()
    assert len(res.fills) == 2 and res.orders["created_session"].nunique() == 1
    assert strat.describe()["params"]["weights"] == {"XNAS_AAPL": 0.6, "XLON_VOD": 0.4}


def test_pairs_strategy_trades_a_planted_cointegrated_pair(catalog):
    from datetime import date as _date

    from quant.core.errors import BacktestError
    from quant.strategies import PairsCointegration

    s, e = _date(2021, 1, 4), _date(2024, 6, 28)
    base_y = make_bars(catalog["XNAS_AAPL"], s, e, seed=81)
    base_x = make_bars(catalog["XNAS_MSFT"], s, e, seed=82)
    n = len(base_x)
    rng = np.random.default_rng(83)
    x_path = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 1.5)
    y_path = 2.0 * x_path + 10.0 + noise

    def _replace(bars, path):
        out = bars.copy()
        factor = path / out["close"].to_numpy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col].to_numpy() * factor
        return out

    bars = pd.concat([_replace(base_y, y_path), _replace(base_x, x_path)])
    strat = PairsCointegration(["XNAS_AAPL", "XNAS_MSFT"], lookback=120, z_window=40, entry_z=1.5, exit_z=0.3)
    cfg = BacktestConfig(
        start=s,
        end=e,
        initial_cash=1_000_000,
        fractional_shares=True,
        allow_short=True,
        max_gross_leverage=1.0,
    )
    res = Backtest(cfg, strat, bars, catalog).run()
    assert len(res.fills) >= 4 and set(res.fills["side"]) == {"buy", "sell"}
    assert (
        res.equity["gross_exposure"].max() <= 1.10
    )  # cap applies at decision time; prices drift until rebalance
    assert (res.events["type"] == "borrow").any()  # shorts cost borrow
    with pytest.raises(BacktestError, match="allow_short"):
        Backtest(
            cfg.model_copy(update={"allow_short": False}),
            PairsCointegration(["XNAS_AAPL", "XNAS_MSFT"], lookback=120, z_window=40, entry_z=1.5),
            bars,
            catalog,
        ).run()
    with pytest.raises(ValueError):
        PairsCointegration(["XNAS_AAPL"])
