from __future__ import annotations

import pytest

from quant.backtest.fills import BarView, fill_price, max_fillable, round_quantity
from quant.backtest.orders import Order, OrderStatus, OrderType, TimeInForce
from quant.core.errors import BacktestError

BAR = BarView(open=100.0, high=105.0, low=96.0, close=102.0, volume=1_000_000)
GAP_UP = BarView(open=110.0, high=112.0, low=108.0, close=111.0, volume=1_000_000)
GAP_DOWN = BarView(open=90.0, high=92.0, low=88.0, close=89.0, volume=1_000_000)


def _o(side, kind="market", **kw):
    return Order("X", side, 10, kind, **kw)


def test_market_fills_at_open():
    assert fill_price(_o("buy"), BAR).price == 100.0
    assert fill_price(_o("sell"), BAR).price == 100.0


def test_limit_buy():
    assert fill_price(_o("buy", "limit", limit_price=101.0), BAR).price == 100.0  # open better than limit
    assert fill_price(_o("buy", "limit", limit_price=98.0), BAR).price == 98.0  # touched intrabar
    assert fill_price(_o("buy", "limit", limit_price=95.0), BAR).price is None  # never reached


def test_limit_sell():
    assert fill_price(_o("sell", "limit", limit_price=99.0), BAR).price == 100.0
    assert fill_price(_o("sell", "limit", limit_price=104.0), BAR).price == 104.0
    assert fill_price(_o("sell", "limit", limit_price=106.0), BAR).price is None


def test_stop_buy_and_sell():
    assert fill_price(_o("buy", "stop", stop_price=103.0), BAR).price == 103.0  # intrabar trigger
    assert fill_price(_o("buy", "stop", stop_price=103.0), GAP_UP).price == 110.0  # gap through: open
    assert fill_price(_o("buy", "stop", stop_price=106.0), BAR).price is None
    assert fill_price(_o("sell", "stop", stop_price=97.0), BAR).price == 97.0
    assert fill_price(_o("sell", "stop", stop_price=97.0), GAP_DOWN).price == 90.0
    assert fill_price(_o("sell", "stop", stop_price=95.0), BAR).price is None


def test_stop_limit_buy_variants():
    # intrabar trigger with limit above stop: fills at the stop
    assert fill_price(_o("buy", "stop_limit", stop_price=103.0, limit_price=104.0), BAR).price == 103.0
    # gap above stop but open within limit: fills at the open
    d = fill_price(_o("buy", "stop_limit", stop_price=103.0, limit_price=111.0), GAP_UP)
    assert d.price == 110.0 and d.triggered and not d.rest_as_limit
    # gap above stop AND above limit: no fill, rests as a limit
    d = fill_price(_o("buy", "stop_limit", stop_price=103.0, limit_price=105.0), GAP_UP)
    assert d.price is None and d.triggered and d.rest_as_limit
    # intrabar trigger with limit below stop: fills at the limit only if the bar reached it
    assert fill_price(_o("buy", "stop_limit", stop_price=103.0, limit_price=97.0), BAR).price == 97.0
    d = fill_price(_o("buy", "stop_limit", stop_price=103.0, limit_price=95.0), BAR)
    assert d.price is None and d.rest_as_limit
    assert fill_price(_o("buy", "stop_limit", stop_price=106.0, limit_price=107.0), BAR).price is None


def test_fill_prices_stay_inside_the_bar():
    for side in ("buy", "sell"):
        for kind, kw in (
            ("market", {}),
            ("limit", {"limit_price": 100.5}),
            ("stop", {"stop_price": 100.5}),
            ("stop_limit", {"stop_price": 100.5, "limit_price": 101.0}),
        ):
            for bar in (BAR, GAP_UP, GAP_DOWN):
                p = fill_price(_o(side, kind, **kw), bar).price
                assert p is None or bar.low <= p <= bar.high


def test_participation_cap_and_rounding():
    assert max_fillable(1_000_000, 500_000, 0.1) == 50_000
    assert max_fillable(100, 500_000, 0.1) == 100
    assert max_fillable(100, 0.0, 0.1) == 100  # no volume information: uncapped
    assert max_fillable(100, 500_000, None) == 100
    assert round_quantity(12.9, 1, False) == 12 and round_quantity(-12.9, 1, False) == -12
    assert round_quantity(12.9, 10, False) == 10 and round_quantity(7, 10, False) == 0
    assert round_quantity(12.9, 1, True) == 12.9


def test_order_validation_and_fill_bookkeeping():
    with pytest.raises(BacktestError, match="qty"):
        Order("X", "buy", 0)
    with pytest.raises(BacktestError, match="limit_price"):
        Order("X", "buy", 1, "limit")
    with pytest.raises(BacktestError, match="stop_price"):
        Order("X", "buy", 1, "stop")
    o = Order("X", "buy", 10, tif="gtc")
    assert o.order_type is OrderType.MARKET and o.tif is TimeInForce.GTC and o.is_open and o.remaining == 10
    o.record_fill(4, 100.0)
    assert o.status is OrderStatus.PARTIAL and o.remaining == 6 and o.avg_fill_price == 100.0
    o.record_fill(6, 110.0)
    assert o.status is OrderStatus.FILLED and o.remaining == 0 and o.avg_fill_price == pytest.approx(106.0)
    with pytest.raises(BacktestError):
        o.record_fill(1, 100.0)
    o.close(OrderStatus.CANCELLED, "late")  # already filled: status unchanged, note kept
    assert o.status is OrderStatus.FILLED and o.notes == ["late"]
    assert o.as_dict()["side"] == "buy"
