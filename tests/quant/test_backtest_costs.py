from __future__ import annotations

from datetime import date

import pytest
from conftest import zero_costs

from quant.backtest.costs import CostModel

ON = date(2024, 7, 1)


def test_uk_sdrt_only_on_buys_of_uk_ordinaries(catalog):
    cm = CostModel(half_spread_bps=0, slippage_bps=0)
    vod, isf, expn = catalog["XLON_VOD"], catalog["XLON_ISF"], catalog["XLON_EXPN"]
    buy = cm.fill_costs(vod, "buy", 20_000, 0.70, ON)  # GBP 14,000
    assert buy.taxes["sdrt"] == pytest.approx(14_000 * 0.005)
    assert buy.taxes["ptm_levy"] == 1.5  # 2024: GBP 1.50 above GBP 10,000
    assert buy.total == pytest.approx(70 + 1.5)
    sell = cm.fill_costs(vod, "sell", 20_000, 0.70, ON)
    assert "sdrt" not in sell.taxes and sell.taxes["ptm_levy"] == 1.5
    assert cm.fill_costs(isf, "buy", 20_000, 8.0, ON).taxes == {}  # ETF: no SDRT, no PTM
    assert cm.fill_costs(expn, "buy", 1_000, 35.0, ON).taxes == {
        "ptm_levy": 1.5
    }  # Jersey-incorporated: exempt


def test_ptm_levy_threshold_and_history(catalog):
    cm = CostModel()
    vod = catalog["XLON_VOD"]
    assert cm.transaction_taxes(vod, "sell", 10_000, 10_000.00, ON) == {}
    assert cm.transaction_taxes(vod, "sell", 10_000, 10_000.01, ON) == {"ptm_levy": 1.5}
    assert cm.transaction_taxes(vod, "sell", 10_000, 10_000.01, date(2020, 1, 1)) == {"ptm_levy": 1.0}
    assert cm.transaction_taxes(vod, "buy", 10, 7.0, date(1980, 1, 1)) == {}  # before every rule


def test_unknown_class_is_taxed_conservatively():
    from quant.core.types import Instrument

    unknown = Instrument(symbol="ZZZ", asset_class="equity", market="XLON", currency="GBP")
    taxes = CostModel().transaction_taxes(unknown, "buy", 100, 1_000.0, ON)
    assert taxes == {"sdrt": pytest.approx(5.0)}


def test_us_sell_side_fees(catalog):
    cm = CostModel(half_spread_bps=0, slippage_bps=0)
    aapl = catalog["XNAS_AAPL"]
    assert cm.fill_costs(aapl, "buy", 100, 200.0, ON).taxes == {}
    sell = cm.fill_costs(aapl, "sell", 100, 200.0, ON).taxes
    assert sell["sec_section31"] == pytest.approx(20_000 * 0.0000278)
    assert sell["finra_taf"] == pytest.approx(100 * 0.000166)
    big = cm.fill_costs(aapl, "sell", 1_000_000, 200.0, ON).taxes
    assert big["finra_taf"] == 8.30  # capped per trade


def test_commission_spread_slippage_and_impact(catalog):
    aapl = catalog["XNAS_AAPL"]
    cm = CostModel(
        commission_per_share=0.005,
        commission_per_order=1.0,
        commission_bps=1.0,
        half_spread_bps=3.0,
        slippage_bps=2.0,
        impact_coeff=5.0,
        apply_transaction_taxes=False,
    )
    c = cm.fill_costs(aapl, "buy", 1_000, 100.0, ON, participation=0.1)
    assert c.commission == pytest.approx(1.0 + 5.0 + 100_000 * 1e-4)
    assert c.spread == pytest.approx(100_000 * 3e-4)
    assert c.slippage == pytest.approx(100_000 * 2e-4 * 1.5)  # impact 1 + 5 x 0.1
    assert c.taxes == {} and c.currency == "USD"
    assert c.total == pytest.approx(c.commission + c.spread + c.slippage)
    cm2 = CostModel(half_spread_bps_by_instrument={"XNAS_AAPL": 10.0}, apply_transaction_taxes=False)
    assert cm2.fill_costs(aapl, "buy", 10, 100.0, ON).spread == pytest.approx(1_000 * 10e-4)


def test_borrow_fx_and_describe():
    cm = CostModel(borrow_bps_annual=252.0, fx_cost_bps=10.0)
    assert cm.daily_borrow_cost(10_000) == pytest.approx(10_000 * 0.0252 / 252)
    assert cm.fx_conversion_cost(-1_000) == pytest.approx(1.0)
    d = cm.describe()
    assert d["fx_cost_bps"] == 10.0 and any("ptm_levy" in r for r in d["unverified_fee_rules"])
    assert zero_costs().describe()["unverified_fee_rules"] == []
