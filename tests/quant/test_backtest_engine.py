from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import ScheduleStrategy, actions_frame, make_bars, plant, zero_costs

from quant.backtest.costs import CostModel
from quant.backtest.engine import Backtest, BacktestConfig
from quant.core.errors import BacktestError
from quant.data.fx import FXRates
from quant.strategies import BuyAndHold

S, E = date(2024, 1, 2), date(2024, 6, 28)


def _cfg(**kw):
    base = dict(
        start=S,
        end=E,
        initial_cash=10_000.0,
        cost_model=zero_costs(),
        fractional_shares=True,
        min_notional=0.0,
    )
    base.update(kw)
    return BacktestConfig(**base)


def _smooth(catalog, iid="XNAS_AAPL", **kw):
    params = dict(vol=0.0, drift=0.001, start_price=100.0)
    params.update(kw)
    return make_bars(catalog[iid], S, E, **params)


def test_buy_and_hold_matches_analytic_return(catalog):
    bars = _smooth(catalog)
    res = Backtest(_cfg(), BuyAndHold(["XNAS_AAPL"]), bars, catalog).run()
    closes = bars["close"].to_numpy()
    sessions = bars["session_date"].dt.date.tolist()
    assert res.equity["equity"].iloc[0] == 10_000.0  # decision after the first close, nothing held yet
    assert len(res.fills) == 1
    fill = res.fills.iloc[0]
    assert fill["session_date"] == sessions[1] and fill["price"] == pytest.approx(
        closes[0]
    )  # open_1 == close_0
    qty = 10_000.0 / closes[0]
    assert fill["qty"] == pytest.approx(qty)
    assert res.equity["equity"].iloc[-1] == pytest.approx(qty * closes[-1])
    assert res.equity["cash_base"].iloc[-1] == pytest.approx(0.0, abs=1e-6)
    assert res.metrics["total_return"] == pytest.approx(closes[-1] / closes[0] - 1)
    assert res.costs == {} and res.variant_count >= 1
    assert any("Universe/survivorship" in c for c in res.caveats)


def test_dividends_are_credited_on_the_ex_date(catalog):
    bars = _smooth(catalog)
    ex = bars["session_date"].iloc[20]
    acts = actions_frame("XNAS_AAPL", [(ex, "dividend", 1.0, "USD")])
    res = Backtest(_cfg(), BuyAndHold(["XNAS_AAPL"]), bars, catalog, actions=acts).run()
    qty = 10_000.0 / bars["close"].iloc[0]
    assert res.equity["equity"].iloc[-1] == pytest.approx(qty * bars["close"].iloc[-1] + qty * 1.0)
    ev = res.events[res.events["type"] == "dividend"]
    assert (
        len(ev) == 1
        and ev["session_date"].iloc[0] == ex.date()
        and ev["amount"].iloc[0] == pytest.approx(qty)
    )
    before = res.equity[res.equity["session_date"] < ex]["cash_base"]
    assert np.allclose(before.iloc[1:], 0.0)


def test_raw_split_and_vendor_adjusted_series_agree(catalog):
    smooth = _smooth(catalog)
    i = 30
    ex = smooth["session_date"].iloc[i]
    raw = plant(smooth, jump=(i, 0.5))  # vendor-raw: prices halve on the ex-date
    acts = actions_frame("XNAS_AAPL", [(ex, "split", 2.0, None)])
    res_raw = Backtest(_cfg(), BuyAndHold(["XNAS_AAPL"]), raw, catalog, actions=acts).run()
    adjusted = make_bars(
        catalog["XNAS_AAPL"], S, E, vol=0.0, drift=0.001, start_price=100.0, adjustment_state="split_adjusted"
    )
    res_adj = Backtest(_cfg(), BuyAndHold(["XNAS_AAPL"]), adjusted, catalog, actions=acts).run()
    np.testing.assert_allclose(res_raw.equity["equity"], res_adj.equity["equity"], rtol=1e-9)
    assert (res_raw.events["type"] == "split").sum() == 1 and (res_adj.events["type"] == "split").sum() == 0
    held_raw = res_raw.positions[res_raw.positions["session_date"] >= ex.date()]["qty"].iloc[0]
    held_adj = res_adj.positions[res_adj.positions["session_date"] >= ex.date()]["qty"].iloc[0]
    assert held_raw == pytest.approx(2 * held_adj)


def test_signal_on_t_fills_at_open_of_t_plus_one(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=11)
    strat = ScheduleStrategy(["XNAS_AAPL"], {5: [("XNAS_AAPL", "buy", 10)]})
    res = Backtest(_cfg(fractional_shares=False), strat, bars, catalog).run()
    fill = res.fills.iloc[0]
    assert fill["session_date"] == bars["session_date"].iloc[6].date()
    assert fill["price"] == pytest.approx(bars["open"].iloc[6])
    assert fill["price"] != pytest.approx(bars["close"].iloc[5])
    order = res.orders.iloc[0]
    assert order["created_session"] == bars["session_date"].iloc[5].date()
    assert order["eligible_from"] == bars["session_date"].iloc[6].date() and order["status"] == "filled"


def test_last_session_decisions_cannot_fill(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=11)
    last = len(bars) - 1
    strat = ScheduleStrategy(["XNAS_AAPL"], {last: [("XNAS_AAPL", "buy", 10)]})
    res = Backtest(_cfg(), strat, bars, catalog).run()
    assert res.orders.empty and res.fills.empty  # no decision on the final session by default
    strat2 = ScheduleStrategy(["XNAS_AAPL"], {last: [("XNAS_AAPL", "buy", 10)]})
    res2 = Backtest(_cfg(decide_on_last_session=True), strat2, bars, catalog).run()
    assert len(res2.orders) == 1 and res2.orders["status"].iloc[0] == "cancelled" and res2.fills.empty


def test_us_uk_holiday_week_no_nan_equity_and_market_specific_fills(catalog):
    s, e = date(2024, 7, 1), date(2024, 7, 8)
    bars = pd.concat(
        [make_bars(catalog["XNAS_AAPL"], s, e, seed=1), make_bars(catalog["XLON_VOD"], s, e, seed=2)]
    )
    fx = FXRates({"GBPUSD": pd.Series(1.27, index=pd.bdate_range(s, e))})
    strat = ScheduleStrategy(
        ["XNAS_AAPL", "XLON_VOD"], {2: [("XNAS_AAPL", "buy", 10), ("XLON_VOD", "buy", 100)]}
    )
    res = Backtest(_cfg(start=s, end=e), strat, bars, catalog, fx=fx).run()
    sessions = res.equity["session_date"].dt.date.tolist()
    assert sessions == [date(2024, 7, d) for d in (1, 2, 3, 4, 5, 8)]  # union calendar includes July 4th
    assert res.equity["equity"].notna().all()
    by_inst = res.fills.set_index("instrument_id")["session_date"]
    assert by_inst["XLON_VOD"] == date(2024, 7, 4)  # London open on US Independence Day
    assert by_inst["XNAS_AAPL"] == date(2024, 7, 5)
    eligible = res.orders.set_index("instrument_id")["eligible_from"]
    assert eligible["XNAS_AAPL"] == date(2024, 7, 5) and eligible["XLON_VOD"] == date(2024, 7, 4)


def test_multi_currency_marks_and_conversions(catalog):
    vod = make_bars(catalog["XLON_VOD"], S, E, seed=3, start_price=0.70, vol=0.01)
    gbpusd = make_bars(catalog["FX_GBPUSD"], S, E, seed=4, start_price=1.27, vol=0.004)
    fx = FXRates.from_bars(gbpusd, catalog)
    strat = ScheduleStrategy(
        ["XLON_VOD"], {1: [("XLON_VOD", "buy", 5_000)], 40: [("XLON_VOD", "sell", 5_000)]}
    )
    cfg = _cfg(initial_cash=10_000.0, cost_model=zero_costs(fx_cost_bps=10.0), fractional_shares=False)
    res = Backtest(cfg, strat, vod, catalog, fx=fx).run()
    assert res.costs["fx_cost"] > 0
    kinds = res.events["type"].tolist()
    assert kinds.count("fx_conversion") == 2  # fund the buy, sweep the sale proceeds
    vod_by_day = vod.assign(session_date=vod["session_date"].dt.date)[["session_date", "close"]]
    held = res.positions.merge(vod_by_day, on="session_date")
    rates = fx.rate("GBP", "USD", pd.DatetimeIndex(held["session_date"]))
    np.testing.assert_allclose(held["value_base"], held["qty"] * held["close"] * rates.to_numpy(), rtol=1e-9)
    assert res.positions["session_date"].max() < res.equity["session_date"].iloc[-1].date()
    assert any("FX rates are vendor daily closes" in c for c in res.caveats)

    cfg_hold = _cfg(initial_cash=10_000.0, cost_model=zero_costs(), cash_policy="hold_foreign")
    strat2 = ScheduleStrategy(
        ["XLON_VOD"], {1: [("XLON_VOD", "buy", 5_000)], 40: [("XLON_VOD", "sell", 5_000)]}
    )
    res2 = Backtest(cfg_hold, strat2, vod, catalog, fx=fx).run()
    assert "fx_conversion" not in res2.events["type"].tolist()


def test_transaction_taxes_end_to_end(catalog):
    s, e = date(2024, 6, 3), date(2024, 7, 31)  # after the first encoded SEC Section 31 rule (2024-05-22)
    vod = make_bars(catalog["XLON_VOD"], s, e, seed=5, start_price=0.70, vol=0.0, drift=0.0)
    isf = make_bars(catalog["XLON_ISF"], s, e, seed=6, start_price=8.0, vol=0.0, drift=0.0)
    aapl = make_bars(catalog["XNAS_AAPL"], s, e, seed=7, start_price=180.0, vol=0.0, drift=0.0)
    fx = FXRates({"GBPUSD": pd.Series(1.25, index=pd.bdate_range(s, e))})
    strat = ScheduleStrategy(
        ["XLON_VOD", "XLON_ISF", "XNAS_AAPL"],
        {
            1: [("XLON_VOD", "buy", 20_000), ("XLON_ISF", "buy", 2_000), ("XNAS_AAPL", "buy", 100)],
            10: [("XLON_VOD", "sell", 20_000), ("XNAS_AAPL", "sell", 100)],
        },
    )
    cm = CostModel(half_spread_bps=0, slippage_bps=0, fx_cost_bps=0)
    res = Backtest(
        _cfg(
            start=s,
            end=e,
            initial_cash=100_000.0,
            cost_model=cm,
            base_currency="USD",
            fractional_shares=False,
        ),
        strat,
        pd.concat([vod, isf, aapl]),
        catalog,
        fx=fx,
    ).run()
    f = res.fills.set_index(["instrument_id", "side"])
    vod_buy = f.loc[("XLON_VOD", "buy")]
    assert vod_buy["taxes"] == pytest.approx(0.005 * vod_buy["gross_notional"] + 1.5)
    assert f.loc[("XLON_VOD", "sell")]["taxes"] == pytest.approx(1.5)  # PTM only
    assert f.loc[("XLON_ISF", "buy")]["taxes"] == 0.0
    assert f.loc[("XNAS_AAPL", "buy")]["taxes"] == 0.0
    aapl_sell = f.loc[("XNAS_AAPL", "sell")]
    assert aapl_sell["taxes"] == pytest.approx(aapl_sell["gross_notional"] * 0.0000278 + 100 * 0.000166)
    assert {"sdrt", "ptm_levy", "sec_section31", "finra_taf"} <= set(res.costs)
    assert any("not yet verified" in c for c in res.caveats)


def test_cash_accounting_identity_and_fills_inside_bars(catalog):
    from quant.strategies import TrendFollowing

    bars = pd.concat(
        [make_bars(catalog["XNAS_AAPL"], S, E, seed=21), make_bars(catalog["XNYS_JPM"], S, E, seed=22)]
    )
    cm = CostModel(
        commission_per_share=0.01, half_spread_bps=3, slippage_bps=2, apply_transaction_taxes=False
    )
    strat = TrendFollowing(["XNAS_AAPL", "XNYS_JPM"], fast=5, slow=20)
    res = Backtest(
        _cfg(initial_cash=100_000.0, cost_model=cm, fractional_shares=False), strat, bars, catalog
    ).run()
    assert len(res.fills) >= 3
    fills = res.fills.copy()
    fills["signed"] = np.where(fills["side"] == "buy", 1.0, -1.0) * fills["gross_notional"]
    fills["flow"] = fills["signed"] + fills["total_cost"]
    flows = fills.groupby("session_date")["flow"].sum()
    expected_cash = (
        100_000.0 - flows.reindex(res.equity["session_date"].dt.date, fill_value=0.0).cumsum().to_numpy()
    )
    np.testing.assert_allclose(res.equity["cash_base"].to_numpy(), expected_cash, rtol=1e-9)
    pos_value = res.positions.groupby("session_date")["value_base"].sum()
    pv = pos_value.reindex(res.equity["session_date"].dt.date, fill_value=0.0).to_numpy()
    np.testing.assert_allclose(res.equity["equity"].to_numpy(), expected_cash + pv, rtol=1e-9)
    merged = fills.merge(
        bars.assign(session_date=bars["session_date"].dt.date), on=["instrument_id", "session_date"]
    )
    assert ((merged["price"] >= merged["low"] - 1e-9) & (merged["price"] <= merged["high"] + 1e-9)).all()
    assert sum(res.costs.values()) == pytest.approx(fills["total_cost"].sum())


def test_participation_cap_gives_partial_fills(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=8)  # volume ~ 1e6 per session
    gtc = ScheduleStrategy(["XNAS_AAPL"], {0: [("XNAS_AAPL", "buy", 250_000, {"tif": "gtc"})]})
    res = Backtest(
        _cfg(initial_cash=1e9, participation_cap=0.1, fractional_shares=False), gtc, bars, catalog
    ).run()
    assert len(res.fills) >= 2 and res.fills["order_id"].nunique() == 1
    assert res.fills["qty"].sum() == pytest.approx(250_000)
    vol = bars.set_index(bars["session_date"].dt.date)["volume"]
    for _, f in res.fills.iloc[:-1].iterrows():
        assert f["qty"] <= 0.1 * vol[f["session_date"]] + 1
    assert res.orders["status"].iloc[0] == "filled"
    day = ScheduleStrategy(["XNAS_AAPL"], {0: [("XNAS_AAPL", "buy", 250_000)]})
    res_day = Backtest(
        _cfg(initial_cash=1e9, participation_cap=0.1, fractional_shares=False), day, bars, catalog
    ).run()
    assert len(res_day.fills) == 1 and res_day.orders["status"].iloc[0] == "expired"
    assert "remainder expired" in res_day.orders["notes"].iloc[0]


def test_cash_long_only_and_leverage_constraints(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=9)
    greedy = ScheduleStrategy(["XNAS_AAPL"], {0: [("XNAS_AAPL", "buy", 10_000_000)]})
    res = Backtest(
        _cfg(initial_cash=10_000.0, participation_cap=None, fractional_shares=False, max_gross_leverage=1e3),
        greedy,
        bars,
        catalog,
    ).run()
    assert (
        res.fills["gross_notional"].iloc[0] <= 10_000.0 and "cash-constrained" in res.orders["notes"].iloc[0]
    )
    naked = ScheduleStrategy(["XNAS_AAPL"], {0: [("XNAS_AAPL", "sell", 10)]})
    res2 = Backtest(_cfg(), naked, bars, catalog).run()
    assert res2.fills.empty and "long-only" in res2.orders["notes"].iloc[0]
    jpm = make_bars(catalog["XNYS_JPM"], S, E, seed=10)
    levered = ScheduleStrategy(
        ["XNAS_AAPL", "XNYS_JPM"], {}, weights_at={0: {"XNAS_AAPL": 0.8, "XNYS_JPM": 0.8}}
    )
    res3 = Backtest(_cfg(fractional_shares=False), levered, pd.concat([bars, jpm]), catalog).run()
    assert res3.equity["gross_exposure"].max() <= 1.0 + 1e-6
    assert res3.orders["notes"].str.contains("scaled for gross leverage").all()


def test_fills_do_not_depend_on_adjustment(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=12)
    ex = bars["session_date"].iloc[25]
    acts = actions_frame("XNAS_AAPL", [(ex, "dividend", 0.5, "USD")])
    runs = []
    for flag in (True, False):
        strat = ScheduleStrategy(
            ["XNAS_AAPL"], {3: [("XNAS_AAPL", "buy", 10)], 50: [("XNAS_AAPL", "sell", 10)]}
        )
        runs.append(Backtest(_cfg(adjust_as_of=flag), strat, bars, catalog, actions=acts).run())
    pd.testing.assert_frame_equal(runs[0].fills, runs[1].fills)
    pd.testing.assert_frame_equal(runs[0].equity, runs[1].equity)


def test_result_summary_and_persistence(tmp_path, catalog):
    bars = _smooth(catalog)
    res = Backtest(_cfg(), BuyAndHold(["XNAS_AAPL"]), bars, catalog).run()
    text = res.summary()
    assert "buy_and_hold" in text and "caveats" in text and "total return" in text
    out = res.to_dir(tmp_path / "run")
    assert (out / "result.json").exists() and (out / "equity.parquet").exists()
    assert len(pd.read_parquet(out / "fills.parquet")) == 1


def test_configuration_errors(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E)
    with pytest.raises(BacktestError, match="unknown instruments"):
        Backtest(_cfg(), BuyAndHold(["XNAS_NOPE"]), bars, catalog)
    with pytest.raises(BacktestError, match="no bars"):
        Backtest(_cfg(), BuyAndHold(["XNYS_JPM"]), bars, catalog)
    with pytest.raises(BacktestError, match="FX rates are required"):
        vod = make_bars(catalog["XLON_VOD"], S, E)
        Backtest(_cfg(), BuyAndHold(["XLON_VOD"]), vod, catalog).run()
