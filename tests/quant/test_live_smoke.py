"""Opt-in live smoke tests: one small request to each free vendor. Never run in CI (`-m live`)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant.core.catalog import load_default_catalog
from quant.data.adapters.boe import BankOfEnglandAdapter
from quant.data.adapters.fred import FREDAdapter
from quant.data.adapters.ons import ONSAdapter
from quant.data.adapters.yfinance_adapter import YFinanceAdapter

pytestmark = pytest.mark.live

END = date.today() - timedelta(days=3)
START = END - timedelta(days=30)


def test_live_yfinance_one_symbol():
    res = YFinanceAdapter().get_bars([load_default_catalog()["XNAS_AAPL"]], START, END)
    assert res.errors == {}, res.errors
    assert len(res.frame) >= 10 and (res.frame["close"] > 0).all()


def test_live_fred_one_series():
    res = FREDAdapter().get_macro_series(["DGS10"], START, END)
    assert res.errors == {}, res.errors
    assert len(res.frame) >= 10


def test_live_boe_one_series():
    res = BankOfEnglandAdapter().get_macro_series(["IUDBEDR"], START, END)
    assert res.errors == {}, res.errors
    assert len(res.frame) >= 10


def test_live_ons_one_series():
    res = ONSAdapter().get_macro_series(["MM23/L55O"], END - timedelta(days=400), END)
    assert res.errors == {}, res.errors
    assert len(res.frame) >= 6


def test_live_wikipedia_constituents_shapes():
    from quant.universe.constituents import load_snapshot

    sp = load_snapshot("sp500")
    assert 480 <= len(sp.members) <= 520
    assert sp.changes is None  # the changes table is gone from the page; reconstruction needs an import
    ftse = load_snapshot("ftse100")
    assert 95 <= len(ftse.members) <= 105


def test_live_alpaca_market_data():
    """Needs Alpaca keys: from QUANT_ALPACA_* or injected by a proxy for data.alpaca.markets."""
    import os

    from quant.data.adapters.alpaca_data import AlpacaDataAdapter

    key = os.environ.get("QUANT_ALPACA_API_KEY", "proxy-injected")
    secret = os.environ.get("QUANT_ALPACA_SECRET_KEY", "proxy-injected")
    cat = load_default_catalog()
    res = AlpacaDataAdapter(key, secret).get_bars([cat["XNAS_AAPL"], cat["XNYS_JPM"]], START, END)
    assert res.errors == {}, res.errors
    assert len(res.frame) >= 10 and (res.frame["adjustment_state"] == "raw").all()
