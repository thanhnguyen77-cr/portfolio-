"""Alpaca PAPER broker adapter (US equities only). Constructed with paper=True; `paper=False` raises
`LiveTradingDisabled` before any client is built. The SDK (`alpaca-py`) is optional: without it, or a
`client_factory`, the adapter raises `AdapterNotConfigured` with the install step.

Fills reported by Alpaca carry no explicit cost breakdown (commission-free venue); spread and slippage are
embedded in the fill price, which this adapter reports as-is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.orders import OrderStatus, OrderType, TimeInForce
from quant.core.errors import AdapterNotConfigured, BrokerError, LiveTradingDisabled
from quant.core.types import Instrument, Market
from quant.paper.broker import AccountSnapshot, BrokerFill, BrokerOrder, OrderRequest, Position

_STATUS = {
    "new": OrderStatus.OPEN,
    "accepted": OrderStatus.OPEN,
    "pending_new": OrderStatus.OPEN,
    "held": OrderStatus.OPEN,
    "partially_filled": OrderStatus.PARTIAL,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
}


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_value(v: Any) -> str:
    return str(getattr(v, "value", v)).lower()


class AlpacaBroker:
    name = "alpaca"
    paper = True
    US_MARKETS = {Market.XNYS.value, Market.XNAS.value}

    def __init__(
        self,
        api_key: str | None,
        secret_key: str | None,
        *,
        paper: bool = True,
        instruments: Mapping[str, Instrument] | None = None,
        base_currency: str = "USD",
        client_factory: Callable[..., Any] | None = None,
        account_id: str = "alpaca-paper",
    ) -> None:
        if not paper:
            raise LiveTradingDisabled(
                "AlpacaBroker is paper-only in this version; live trading is disabled by design"
            )
        if base_currency != "USD":
            raise BrokerError("Alpaca accounts are USD-denominated")
        self.api_key, self.secret_key = api_key, secret_key
        self.instruments = dict(instruments or {})
        self.base_currency = base_currency
        self.account_id = account_id
        self._client_factory = client_factory
        self._client: Any = None

    # ------------------------------------------------------------------ client
    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key or not self.secret_key:
            raise AdapterNotConfigured(
                "Alpaca paper trading needs QUANT_ALPACA_API_KEY and QUANT_ALPACA_SECRET_KEY in the "
                "environment"
            )
        if self._client_factory is not None:
            self._client = self._client_factory(self.api_key, self.secret_key, paper=True)
            return self._client
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover - exercised only without the optional extra
            raise AdapterNotConfigured(
                "alpaca-py is not installed: pip install 'quant-platform[alpaca]'"
            ) from exc
        self._client = TradingClient(self.api_key, self.secret_key, paper=True)
        return self._client

    def _symbol(self, instrument_id: str) -> str:
        inst = self.instruments.get(instrument_id)
        if inst is None:
            raise BrokerError(f"unknown instrument {instrument_id!r}")
        if inst.market.value not in self.US_MARKETS:
            raise BrokerError(
                f"Alpaca routes US equities only; {instrument_id} trades on {inst.market.value}"
            )
        return inst.identifiers.get("alpaca") or inst.symbol

    def _instrument_for_symbol(self, symbol: str) -> str:
        for iid, inst in self.instruments.items():
            if (inst.identifiers.get("alpaca") or inst.symbol) == symbol:
                return iid
        return symbol

    @staticmethod
    def _order_data(request: OrderRequest, symbol: str) -> Any:
        """SDK request object when alpaca-py is importable, else a plain dict with the same fields."""
        payload = {
            "symbol": symbol,
            "qty": request.qty,
            "side": request.side.value,
            "type": request.order_type.value,
            "time_in_force": request.tif.value,
            "limit_price": request.limit_price,
            "stop_price": request.stop_price,
            "client_order_id": request.client_order_id,
        }
        try:
            from alpaca.trading.enums import OrderSide
            from alpaca.trading.enums import TimeInForce as TIF
            from alpaca.trading.requests import (
                LimitOrderRequest,
                MarketOrderRequest,
                StopLimitOrderRequest,
                StopOrderRequest,
            )
        except ImportError:
            return payload
        common = {
            "symbol": symbol,
            "qty": request.qty,
            "side": OrderSide(request.side.value),
            "time_in_force": TIF(request.tif.value),
            "client_order_id": request.client_order_id,
        }
        if request.order_type is OrderType.MARKET:
            return MarketOrderRequest(**common)
        if request.order_type is OrderType.LIMIT:
            return LimitOrderRequest(limit_price=request.limit_price, **common)
        if request.order_type is OrderType.STOP:
            return StopOrderRequest(stop_price=request.stop_price, **common)
        return StopLimitOrderRequest(limit_price=request.limit_price, stop_price=request.stop_price, **common)

    # ------------------------------------------------------------------ Broker protocol
    def account(self) -> AccountSnapshot:
        a = self._get_client().get_account()
        cash = float(_get(a, "cash", 0.0) or 0.0)
        equity = float(_get(a, "equity", 0.0) or 0.0)
        last_equity = _get(a, "last_equity")
        pv = float(_get(a, "long_market_value", 0.0) or 0.0) + float(
            _get(a, "short_market_value", 0.0) or 0.0
        )
        return AccountSnapshot(
            account_id=str(_get(a, "account_number", self.account_id)),
            broker=self.name,
            as_of=date.today(),
            base_currency=self.base_currency,
            cash={self.base_currency: cash},
            cash_base=cash,
            positions_value=pv,
            equity=equity,
            last_equity=float(last_equity) if last_equity is not None else None,
            n_positions=len(self.positions()),
            n_open_orders=len(self.open_orders()),
        )

    def positions(self) -> list[Position]:
        out = []
        for p in self._get_client().get_all_positions():
            qty = float(_get(p, "qty", 0.0))
            if _enum_value(_get(p, "side", "long")) == "short":
                qty = -abs(qty)
            price = _get(p, "current_price")
            out.append(
                Position(
                    self._instrument_for_symbol(str(_get(p, "symbol"))),
                    qty,
                    float(_get(p, "avg_entry_price", 0.0) or 0.0),
                    self.base_currency,
                    float(price) if price is not None else None,
                    float(_get(p, "market_value", 0.0) or 0.0),
                )
            )
        return out

    def _to_order(self, o: Any) -> BrokerOrder:
        status = _STATUS.get(_enum_value(_get(o, "status", "new")), OrderStatus.OPEN)
        filled = float(_get(o, "filled_qty", 0.0) or 0.0)
        avg = _get(o, "filled_avg_price")
        created = _get(o, "created_at")
        return BrokerOrder(
            order_id=str(_get(o, "id")),
            instrument_id=self._instrument_for_symbol(str(_get(o, "symbol"))),
            side=_enum_value(_get(o, "side", "buy")),
            qty=float(_get(o, "qty", 0.0) or 0.0),
            order_type=_enum_value(_get(o, "type", _get(o, "order_type", "market"))),
            limit_price=float(_get(o, "limit_price")) if _get(o, "limit_price") is not None else None,
            stop_price=float(_get(o, "stop_price")) if _get(o, "stop_price") is not None else None,
            tif=_enum_value(_get(o, "time_in_force", "day")),
            status=status,
            filled_qty=filled,
            avg_fill_price=float(avg) if avg is not None else None,
            created_session=pd.Timestamp(created).date() if created is not None else None,
            client_order_id=_get(o, "client_order_id"),
        )

    def orders(self) -> list[BrokerOrder]:
        return [self._to_order(o) for o in self._get_client().get_orders()]

    def open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.orders() if o.is_open]

    def fills(self) -> list[BrokerFill]:
        out = []
        for o in self.orders():
            if o.filled_qty > 0 and o.avg_fill_price is not None:
                out.append(
                    BrokerFill(
                        f"{o.order_id}-fill",
                        o.order_id,
                        o.instrument_id,
                        o.side,
                        o.filled_qty,
                        o.avg_fill_price,
                        o.created_session or date.today(),
                        self.base_currency,
                        tag=o.tag,
                    )
                )
        return out

    def submit(self, request: OrderRequest) -> BrokerOrder:
        symbol = self._symbol(request.instrument_id)
        if request.tif not in (TimeInForce.DAY, TimeInForce.GTC):
            raise BrokerError("Alpaca adapter supports day and gtc orders")
        raw = self._get_client().submit_order(order_data=self._order_data(request, symbol))
        order = self._to_order(raw)
        order.tag = request.tag
        return order

    def cancel(self, order_id: str) -> BrokerOrder:
        client = self._get_client()
        client.cancel_order_by_id(order_id)
        raw = client.get_order_by_id(order_id)
        return self._to_order(raw)
