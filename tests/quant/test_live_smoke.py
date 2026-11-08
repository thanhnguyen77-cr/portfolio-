"""Opt-in live smoke tests: one small request to each free vendor. Never run in CI (`-m live`)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant.core.catalog import load_default_catalog
from quant.data.adapters.boe import BankOfEnglandAdapter
from quant.data.adapters.fred import FREDAdapter
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
