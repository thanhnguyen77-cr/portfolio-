"""Broker contract shared by the built-in SimBroker and the optional Alpaca paper adapter.

Everything here is paper trading. `Broker.paper` is always True in this version; the only live path in the
package is `quant.config.live_gate.require_live_confirmation`, which raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

from quant.backtest.orders import Order, OrderStatus, OrderType, TimeInForce
from quant.core.errors import BrokerError
from quant.core.types import Side


@dataclass
class OrderRequest:
    instrument_id: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    tag: str | None = None
    client_order_id: str | None = None

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.order_type = OrderType(self.order_type)
        self.tif = TimeInForce(self.tif)
        if not self.qty > 0:
            raise BrokerError(f"order qty must be > 0, got {self.qty}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and not (self.limit_price or 0) > 0:
            raise BrokerError(f"{self.order_type.value} order needs a positive limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and not (self.stop_price or 0) > 0:
            raise BrokerError(f"{self.order_type.value} order needs a positive stop_price")

    def to_order(self) -> Order:
        return Order(
            self.instrument_id,
            self.side,
            self.qty,
            self.order_type,
            limit_price=self.limit_price,
            stop_price=self.stop_price,
            tif=self.tif,
            tag=self.tag,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif.value,
            "tag": self.tag,
            "client_order_id": self.client_order_id,
        }


@dataclass
class BrokerOrder:
    """An order as the broker knows it. Attribute names match `quant.backtest.orders.Order` so the engine's
    fill rules (`fill_price`) apply unchanged."""

    order_id: str
    instrument_id: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    tif: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.OPEN
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    created_session: date | None = None
    eligible_from: date | None = None
    tag: str | None = None
    client_order_id: str | None = None
    triggered: bool = False
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.side = Side(self.side)
        self.order_type = OrderType(self.order_type)
        self.tif = TimeInForce(self.tif)
        self.status = OrderStatus(self.status)

    @property
    def remaining(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIAL)

    def record_fill(self, qty: float, price: float) -> None:
        if qty <= 0 or qty > self.remaining + 1e-12:
            raise BrokerError(
                f"invalid fill qty {qty} for order {self.order_id} (remaining {self.remaining})"
            )
        prev = (self.avg_fill_price or 0.0) * self.filled_qty
        self.filled_qty += qty
        self.avg_fill_price = (prev + qty * price) / self.filled_qty
        self.status = OrderStatus.FILLED if self.remaining <= 1e-12 else OrderStatus.PARTIAL

    def close(self, status: OrderStatus, note: str | None = None) -> None:
        if self.is_open:
            self.status = status
        if note:
            self.notes.append(note)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "instrument_id": self.instrument_id,
            "side": self.side.value,
            "qty": self.qty,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.tif.value,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "created_session": self.created_session,
            "eligible_from": self.eligible_from,
            "tag": self.tag,
            "client_order_id": self.client_order_id,
            "triggered": self.triggered,
            "notes": "; ".join(self.notes),
        }


@dataclass
class BrokerFill:
    fill_id: str
    order_id: str
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

    def __post_init__(self) -> None:
        self.side = Side(self.side)

    @property
    def gross_notional(self) -> float:
        return self.qty * self.price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread + self.slippage + self.taxes + self.fx_cost

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side is Side.BUY else -self.qty

    def as_dict(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
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


@dataclass
class Position:
    instrument_id: str
    qty: float
    avg_price: float
    currency: str
    last_price: float | None = None
    market_value_base: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "qty": self.qty,
            "avg_price": self.avg_price,
            "currency": self.currency,
            "last_price": self.last_price,
            "market_value_base": self.market_value_base,
            "unrealised_local": (
                (self.last_price - self.avg_price) * self.qty if self.last_price is not None else None
            ),
        }


@dataclass
class AccountSnapshot:
    account_id: str
    broker: str
    as_of: date | None
    base_currency: str
    cash: dict[str, float]
    cash_base: float
    positions_value: float
    equity: float
    last_equity: float | None
    n_positions: int
    n_open_orders: int
    paper: bool = True

    @property
    def day_change_pct(self) -> float | None:
        if self.last_equity:
            return (self.equity / self.last_equity - 1.0) * 100.0
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "broker": self.broker,
            "paper": self.paper,
            "as_of": self.as_of,
            "base_currency": self.base_currency,
            "cash": dict(self.cash),
            "cash_base": self.cash_base,
            "positions_value": self.positions_value,
            "equity": self.equity,
            "last_equity": self.last_equity,
            "day_change_pct": self.day_change_pct,
            "n_positions": self.n_positions,
            "n_open_orders": self.n_open_orders,
        }


@runtime_checkable
class Broker(Protocol):
    name: str
    paper: bool

    def account(self) -> AccountSnapshot: ...

    def positions(self) -> list[Position]: ...

    def orders(self) -> list[BrokerOrder]: ...

    def open_orders(self) -> list[BrokerOrder]: ...

    def fills(self) -> list[BrokerFill]: ...

    def submit(self, request: OrderRequest) -> BrokerOrder: ...

    def cancel(self, order_id: str) -> BrokerOrder: ...
