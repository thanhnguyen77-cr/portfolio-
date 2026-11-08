"""Strategy contract and the per-session Context the engine hands to it.

A strategy sees only data up to and including the current session (adjusted as of that session) and may
return explicit orders, target weights (converted to market orders by the engine), or nothing. Orders are
eligible from the next session of the instrument's own calendar; there is no same-session fill.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from quant.backtest.fills import round_quantity
from quant.backtest.orders import Order, OrderType, TimeInForce
from quant.core.errors import BacktestError
from quant.core.types import Instrument, Side


@dataclass
class Context:
    session_date: date
    instruments: Mapping[str, Instrument]
    base_currency: str
    positions: Mapping[str, float]
    cash: Mapping[str, float]
    equity: float
    allow_short: bool
    fractional_shares: bool
    min_notional: float
    _history: Callable[[str, int | None], pd.DataFrame]
    _close: Callable[[str], float | None]
    _fx: Callable[[str, str], float]
    session_index: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ data access (never beyond t)
    def history(self, instrument_id: str, n: int | None = None) -> pd.DataFrame:
        """Bars with session_date <= today, adjusted as of today (adj_* columns), last `n` rows if given."""
        return self._history(instrument_id, n)

    def close(self, instrument_id: str) -> float | None:
        """Latest close at or before today in the instrument's own currency (None if never traded)."""
        return self._close(instrument_id)

    def fx(self, from_ccy: str, to_ccy: str) -> float:
        return self._fx(from_ccy, to_ccy)

    def position_value(self, instrument_id: str) -> float:
        qty = self.positions.get(instrument_id, 0.0)
        px = self.close(instrument_id)
        if not qty or px is None:
            return 0.0
        return qty * px * self.fx(self.instruments[instrument_id].currency.value, self.base_currency)

    def weight(self, instrument_id: str) -> float:
        return self.position_value(instrument_id) / self.equity if self.equity else 0.0

    # ------------------------------------------------------------------ order helpers
    def market_order(self, instrument_id: str, side: Side | str, qty: float, *, tif="day", tag=None) -> Order:  # noqa: ANN001
        return Order(instrument_id, Side(side), qty, OrderType.MARKET, tif=TimeInForce(tif), tag=tag)

    def limit_order(
        self, instrument_id: str, side: Side | str, qty: float, limit_price: float, *, tif="day", tag=None
    ) -> Order:  # noqa: ANN001,E501
        return Order(
            instrument_id,
            Side(side),
            qty,
            OrderType.LIMIT,
            limit_price=limit_price,
            tif=TimeInForce(tif),
            tag=tag,
        )

    def stop_order(
        self, instrument_id: str, side: Side | str, qty: float, stop_price: float, *, tif="gtc", tag=None
    ) -> Order:  # noqa: ANN001,E501
        return Order(
            instrument_id,
            Side(side),
            qty,
            OrderType.STOP,
            stop_price=stop_price,
            tif=TimeInForce(tif),
            tag=tag,
        )

    def target_weights(self, weights: Mapping[str, float], *, tag: str | None = None) -> list[Order]:
        """Market orders moving the portfolio toward `weights` (fractions of current equity in base ccy)."""
        orders: list[Order] = []
        for iid, w in weights.items():
            if iid not in self.instruments:
                raise BacktestError(f"target weight for unknown instrument {iid!r}")
            if w < 0 and not self.allow_short:
                raise BacktestError(f"negative target weight for {iid!r} but allow_short is False")
            px = self.close(iid)
            if px is None or not px > 0:
                continue
            inst = self.instruments[iid]
            px_base = px * self.fx(inst.currency.value, self.base_currency)
            desired = w * self.equity / px_base
            delta = round_quantity(
                desired - self.positions.get(iid, 0.0), inst.lot_size, self.fractional_shares
            )
            if delta == 0 or abs(delta) * px_base < self.min_notional:
                continue
            orders.append(self.market_order(iid, Side.BUY if delta > 0 else Side.SELL, abs(delta), tag=tag))
        return orders

    def target_quantities(self, quantities: Mapping[str, float], *, tag: str | None = None) -> list[Order]:
        orders: list[Order] = []
        for iid, q in quantities.items():
            delta = q - self.positions.get(iid, 0.0)
            if abs(delta) > 1e-12:
                orders.append(
                    self.market_order(iid, Side.BUY if delta > 0 else Side.SELL, abs(delta), tag=tag)
                )
        return orders


class Strategy:
    """Base class. Subclasses set `name`, optionally `warmup`, and implement `on_session`."""

    name: str = "strategy"
    warmup: int = 0

    def __init__(self, instruments: Sequence[str], **params: Any) -> None:
        self.instruments = list(instruments)
        self.params: dict[str, Any] = dict(params)

    def on_start(self, instruments: Mapping[str, Instrument]) -> None:
        """Called once before the first session."""

    def on_session(self, ctx: Context) -> list[Order] | Mapping[str, float] | None:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "instruments": list(self.instruments), "params": dict(self.params)}
