"""Fill rules against the first bar after submission (conservative, matching the legacy backtester):

  market      fills at the open
  limit  buy  open if open <= limit, else limit if low <= limit        (sell: mirror with high)
  stop   buy  triggers if high >= stop; gap above stop fills at the open, else at the stop (sell: mirror)
  stop-limit  triggers like a stop; a gap beyond the stop fills at the open only if the open respects the
              limit, otherwise the order rests as a plain limit; an intrabar trigger fills at the stop when
              the limit allows it, else at the limit if the bar reached it

Every fill price is inside [low, high] by construction. Quantity is capped by `participation_cap` x bar
volume (partial fills); bars with zero/unknown volume (FX, indices) are uncapped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quant.backtest.orders import Order, OrderType
from quant.core.types import Side


@dataclass(frozen=True)
class BarView:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FillDecision:
    price: float | None  # None: no fill on this bar
    triggered: bool = False  # a stop was touched (relevant for stop-limit resting as a limit)
    rest_as_limit: bool = False


def fill_price(order: Order, bar: BarView) -> FillDecision:
    o, buy = order, order.side is Side.BUY
    if o.order_type is OrderType.MARKET:
        return FillDecision(bar.open)
    if o.order_type is OrderType.LIMIT:
        lim = float(o.limit_price)  # type: ignore[arg-type]
        if buy:
            if bar.open <= lim:
                return FillDecision(bar.open)
            return FillDecision(lim if bar.low <= lim else None)
        if bar.open >= lim:
            return FillDecision(bar.open)
        return FillDecision(lim if bar.high >= lim else None)
    stop = float(o.stop_price)  # type: ignore[arg-type]
    touched = bar.high >= stop if buy else bar.low <= stop
    if not touched and not o.triggered:
        return FillDecision(None)
    gap = (bar.open >= stop) if buy else (bar.open <= stop)
    if o.order_type is OrderType.STOP:
        return FillDecision(bar.open if gap else stop, triggered=True)
    lim = float(o.limit_price)  # type: ignore[arg-type]
    if o.triggered:  # already a resting limit
        return fill_price(_as_limit(o), bar)
    if gap:
        ok = bar.open <= lim if buy else bar.open >= lim
        return FillDecision(bar.open if ok else None, triggered=True, rest_as_limit=not ok)
    if (lim >= stop) if buy else (lim <= stop):
        return FillDecision(stop, triggered=True)
    reached = bar.low <= lim if buy else bar.high >= lim
    return FillDecision(lim if reached else None, triggered=True, rest_as_limit=not reached)


def _as_limit(order: Order) -> Order:
    clone = Order(
        instrument_id=order.instrument_id,
        side=order.side,
        qty=order.qty,
        order_type=OrderType.LIMIT,
        limit_price=order.limit_price,
        tif=order.tif,
    )
    return clone


def max_fillable(remaining: float, volume: float, participation_cap: float | None) -> float:
    if participation_cap is None or not volume or not math.isfinite(volume) or volume <= 0:
        return remaining
    return min(remaining, participation_cap * volume)


def round_quantity(qty: float, lot_size: int, fractional: bool) -> float:
    """Round toward zero to a whole multiple of the lot size (unless fractional shares are allowed)."""
    if fractional:
        return qty
    lots = math.floor(abs(qty) / lot_size + 1e-9)
    return math.copysign(lots * lot_size, qty) if lots else 0.0
