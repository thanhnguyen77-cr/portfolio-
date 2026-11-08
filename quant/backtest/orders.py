"""Orders and fills."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

from quant.core.errors import BacktestError
from quant.core.types import Side


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"  # valid for the first session of the instrument's calendar after submission
    GTC = "gtc"  # rests until filled or cancelled


class OrderStatus(StrEnum):
    OPEN = "open"
    PARTIAL = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class Order:
    instrument_id: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    tag: str | None = None
    order_id: int | None = None
    created_session: date | None = None
    eligible_from: date | None = None
    status: OrderStatus = OrderStatus.OPEN
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    triggered: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.order_type = OrderType(self.order_type)
        self.tif = TimeInForce(self.tif)
        if not self.qty > 0:
            raise BacktestError(f"order qty must be > 0, got {self.qty}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not (self.limit_price or 0) > 0:
            raise BacktestError(f"{self.order_type.value} order needs a positive limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not (self.stop_price or 0) > 0:
            raise BacktestError(f"{self.order_type.value} order needs a positive stop_price")

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side is Side.BUY else -self.qty

    def record_fill(self, qty: float, price: float) -> None:
        if qty <= 0 or qty > self.remaining + 1e-12:
            raise BacktestError(
                f"invalid fill qty {qty} for order {self.order_id} (remaining {self.remaining})"
            )
        prev_notional = (self.avg_fill_price or 0.0) * self.filled_qty
        self.filled_qty += qty
        self.avg_fill_price = (prev_notional + qty * price) / self.filled_qty
        self.status = OrderStatus.FILLED if self.remaining <= 1e-12 else OrderStatus.PARTIAL

    def close(self, status: OrderStatus, note: str | None = None) -> None:
        if self.is_open:
            self.status = status
        if note:
            self.notes.append(note)

    def as_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif.value,
            "tag": self.tag,
            "created_session": self.created_session,
            "eligible_from": self.eligible_from,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "notes": "; ".join(self.notes),
        }


@dataclass(frozen=True)
class Fill:
    order_id: int | None
    instrument_id: str
    side: Side
    qty: float
    price: float
    session_date: date
    currency: str
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    taxes: float = 0.0
    fx_cost: float = 0.0
    tag: str | None = None

    @property
    def gross_notional(self) -> float:
        return self.qty * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread + self.slippage + self.taxes + self.fx_cost

    def as_dict(self) -> dict[str, object]:
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "qty": self.qty,
            "price": self.price,
            "session_date": self.session_date,
            "currency": self.currency,
            "gross_notional": self.gross_notional,
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "taxes": self.taxes,
            "fx_cost": self.fx_cost,
            "total_cost": self.total_cost,
            "tag": self.tag,
        }
