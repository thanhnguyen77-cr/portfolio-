"""SimBroker: the default paper broker for every market, multi-currency, with the engine's fill rules.

It re-uses the engine's primitives exactly - `fill_price`, `max_fillable`, `round_quantity` and the shared
`CostModel` - and applies them in the same order (corporate actions, fills, borrow, mark to market), so a
test can pin SimBroker fills to engine fills on identical orders. State lives in an `AccountState` and is
persisted through a `Ledger` after every mutation when one is attached. Unlike the engine, which trims a
buy to the cash available at fill time, the broker rejects an order at submission when the buying power
(cash in base currency, less pending buys) does not cover it - the way a real broker would.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date

import pandas as pd

from quant.backtest.costs import CostModel
from quant.backtest.fills import BarView, fill_price, max_fillable, round_quantity
from quant.backtest.orders import OrderStatus, OrderType, TimeInForce
from quant.core.calendars import get_calendar
from quant.core.errors import BrokerError
from quant.core.time import to_date
from quant.core.types import AdjustmentState, Instrument, Side
from quant.data.fx import FXRates
from quant.paper.broker import AccountSnapshot, BrokerFill, BrokerOrder, OrderRequest, Position
from quant.paper.ledger import AccountState, Ledger, PositionState


class SimBroker:
    name = "sim"
    paper = True

    def __init__(
        self,
        state: AccountState,
        instruments: Mapping[str, Instrument],
        *,
        cost_model: CostModel | None = None,
        fx: FXRates | None = None,
        participation_cap: float | None = 0.10,
        fractional_shares: bool = False,
        allow_short: bool = False,
        cash_policy: str = "convert_at_fill",
        adjustment_state: Mapping[str, str] | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.state = state
        self.instruments = dict(instruments)
        self.cost_model = cost_model or CostModel()
        self.fx = fx
        self.participation_cap = participation_cap
        self.fractional_shares = fractional_shares
        self.allow_short = allow_short
        if cash_policy not in {"convert_at_fill", "hold_foreign"}:
            raise BrokerError("cash_policy must be convert_at_fill or hold_foreign")
        self.cash_policy = cash_policy
        self.adjustment_state = dict(adjustment_state or {})
        self.ledger = ledger

    # ------------------------------------------------------------------ helpers
    @property
    def base(self) -> str:
        return self.state.base_currency

    def _persist(self) -> None:
        if self.ledger is not None:
            self.ledger.save(self.state)

    def _fx(self, from_ccy: str, to_ccy: str, on: date | None) -> float:
        if from_ccy == to_ccy:
            return 1.0
        if self.fx is None:
            raise BrokerError(f"FX rates are required to convert {from_ccy}->{to_ccy}")
        return self.fx.latest(from_ccy, to_ccy, on or self.state.as_of or date.today())

    def _instrument(self, instrument_id: str) -> Instrument:
        inst = self.instruments.get(instrument_id)
        if inst is None:
            raise BrokerError(f"unknown instrument {instrument_id!r} for this broker")
        return inst

    def _cash_base(self, on: date | None) -> float:
        return sum(self._fx(ccy, self.base, on) * amt for ccy, amt in self.state.cash.items() if amt)

    def _positions_value(self, on: date | None) -> tuple[float, float, float]:
        total = gross = net = 0.0
        for iid, p in self.state.positions.items():
            if not p.qty or iid not in self.state.marks:
                continue
            v = p.qty * self.state.marks[iid] * self._fx(p.currency, self.base, on)
            total += v
            gross += abs(v)
            net += v
        return total, gross, net

    def _pending_buy_notional_base(self, on: date | None) -> float:
        total = 0.0
        for o in self.state.orders.values():
            if o.is_open and o.side is Side.BUY:
                inst = self.instruments[o.instrument_id]
                ref = o.limit_price or o.stop_price or self.state.marks.get(o.instrument_id)
                if ref:
                    total += o.remaining * ref * self._fx(inst.currency.value, self.base, on)
        return total

    def _reserved_sell_qty(self, instrument_id: str) -> float:
        return sum(
            o.remaining
            for o in self.state.orders.values()
            if o.is_open and o.side is Side.SELL and o.instrument_id == instrument_id
        )

    # ------------------------------------------------------------------ Broker protocol
    def account(self) -> AccountSnapshot:
        st = self.state
        on = st.as_of
        cash_base = self._cash_base(on)
        pv, _, _ = self._positions_value(on)
        rows = st.equity
        last_equity = float(rows[-2]["equity"]) if len(rows) >= 2 else None
        return AccountSnapshot(
            account_id=st.account_id,
            broker=self.name,
            as_of=on,
            base_currency=self.base,
            cash=dict(st.cash),
            cash_base=cash_base,
            positions_value=pv,
            equity=cash_base + pv,
            last_equity=last_equity,
            n_positions=sum(1 for p in st.positions.values() if abs(p.qty) > 1e-12),
            n_open_orders=sum(1 for o in st.orders.values() if o.is_open),
        )

    def positions(self) -> list[Position]:
        out = []
        for iid, p in self.state.positions.items():
            if abs(p.qty) <= 1e-12:
                continue
            mark = self.state.marks.get(iid)
            mv = p.qty * mark * self._fx(p.currency, self.base, self.state.as_of) if mark else None
            out.append(Position(iid, p.qty, p.avg_price, p.currency, mark, mv))
        return out

    def orders(self) -> list[BrokerOrder]:
        return list(self.state.orders.values())

    def open_orders(self) -> list[BrokerOrder]:
        return [o for o in self.state.orders.values() if o.is_open]

    def fills(self) -> list[BrokerFill]:
        return list(self.state.fills)

    def submit(self, request: OrderRequest) -> BrokerOrder:
        """Validate and accept (status open, eligible from the next session) or reject (status rejected)."""
        st = self.state
        inst = self._instrument(request.instrument_id)
        order = BrokerOrder(
            order_id=st.next_order_id(),
            instrument_id=request.instrument_id,
            side=request.side,
            qty=float(request.qty),
            order_type=request.order_type,
            limit_price=request.limit_price,
            stop_price=request.stop_price,
            tif=request.tif,
            created_session=st.as_of,
            tag=request.tag,
            client_order_id=request.client_order_id,
        )
        reason = self._validate(order, inst)
        if reason:
            order.status = OrderStatus.REJECTED
            order.notes.append(reason)
            st.add_event(
                "order_rejected", st.as_of, f"{order.order_id}: {reason}", instrument_id=inst.instrument_id
            )
        else:
            if st.as_of is None:
                raise BrokerError("broker has no current session; process a session before submitting orders")
            order.eligible_from = get_calendar(inst.calendar_code).next_session(st.as_of)
            st.add_event(
                "order_submitted",
                st.as_of,
                f"{order.order_id} {order.side.value} {order.qty:g} {inst.instrument_id} "
                f"{order.order_type.value}",
                instrument_id=inst.instrument_id,
            )
        st.orders[order.order_id] = order
        self._persist()
        return order

    def _validate(self, order: BrokerOrder, inst: Instrument) -> str | None:
        st = self.state
        if not inst.tradable:
            return f"{inst.instrument_id} is not tradable (analysis-only series)"
        if not self.fractional_shares:
            lots = order.qty / inst.lot_size
            if abs(lots - round(lots)) > 1e-9:
                return f"qty {order.qty:g} is not a whole multiple of lot size {inst.lot_size}"
        ref = order.limit_price or order.stop_price or st.marks.get(inst.instrument_id)
        if not ref or not ref > 0:
            return "no reference price for the instrument yet (no mark and no limit/stop price)"
        on = st.as_of
        if order.side is Side.SELL and not self.allow_short:
            held = st.positions.get(inst.instrument_id, PositionState(0.0, 0.0, inst.currency.value)).qty
            free = held - self._reserved_sell_qty(inst.instrument_id)
            if order.qty > free + 1e-9:
                return f"long-only: sell {order.qty:g} exceeds free quantity {free:g} (held {held:g})"
        if order.side is Side.BUY:
            notional = order.qty * ref * self._fx(inst.currency.value, self.base, on)
            available = self._cash_base(on) - self._pending_buy_notional_base(on)
            if notional > available + 1e-6:
                return (
                    f"insufficient buying power: need {notional:,.2f} {self.base}, available {available:,.2f}"
                )
        return None

    def cancel(self, order_id: str) -> BrokerOrder:
        order = self.state.orders.get(order_id)
        if order is None:
            raise BrokerError(f"order {order_id!r} not found")
        if not order.is_open:
            raise BrokerError(f"order {order_id} is {order.status.value} and cannot be cancelled")
        order.close(OrderStatus.CANCELLED, "cancelled by user")
        self.state.add_event("order_cancelled", self.state.as_of, order_id, instrument_id=order.instrument_id)
        self._persist()
        return order

    # ------------------------------------------------------------------ session processing
    def process_session(
        self,
        session_date: date | str,
        bars: Mapping[str, BarView],
        actions: pd.DataFrame | None = None,
    ) -> list[BrokerFill]:
        """Apply one completed session: corporate actions, fills of eligible orders, borrow, marks."""
        st = self.state
        t = to_date(session_date)
        if st.as_of is not None and t <= st.as_of:
            raise BrokerError(f"session {t} is not after the account's current session {st.as_of}")
        base = self.base
        new_fills: list[BrokerFill] = []
        turnover_base = 0.0

        def convert(amount_from: float, from_ccy: str, to_ccy: str, reason: str) -> float:
            if from_ccy == to_ccy or amount_from <= 0:
                return amount_from
            rate = self._fx(from_ccy, to_ccy, t)
            fee = self.cost_model.fx_conversion_cost(amount_from)
            credited = (amount_from - fee) * rate
            st.cash[from_ccy] = st.cash.get(from_ccy, 0.0) - amount_from
            st.cash[to_ccy] = st.cash.get(to_ccy, 0.0) + credited
            st.add_event("fx_out", t, reason, amount=-amount_from, currency=from_ccy)
            st.add_event("fx_in", t, f"{reason} (fee {fee:.4f} {from_ccy})", amount=credited, currency=to_ccy)
            return credited

        # 1. corporate actions on positions held at the previous close
        if actions is not None and not actions.empty:
            a = actions.copy()
            a["ex_date"] = pd.to_datetime(a["ex_date"]).dt.normalize()
            todays = a[a["ex_date"] == pd.Timestamp(t)]
            for iid, kind, value, ccy in todays[
                ["instrument_id", "action_type", "value", "currency"]
            ].itertuples(index=False):
                pos = st.positions.get(iid)
                if pos is None or not pos.qty or iid not in self.instruments:
                    continue
                inst = self.instruments[iid]
                if (
                    kind == "split"
                    and value > 0
                    and self.adjustment_state.get(iid) == AdjustmentState.RAW.value
                ):
                    pos.qty *= float(value)
                    pos.avg_price /= float(value)
                    st.add_event(
                        "split", t, f"ratio {value} pre-fill", instrument_id=iid, amount=float(value)
                    )
                elif kind == "dividend" and value > 0:
                    div_ccy = ccy if isinstance(ccy, str) and ccy else inst.currency.value
                    amount = pos.qty * float(value)
                    st.cash[div_ccy] = st.cash.get(div_ccy, 0.0) + amount
                    st.add_event(
                        "dividend",
                        t,
                        f"{value} per share",
                        instrument_id=iid,
                        amount=amount,
                        currency=div_ccy,
                    )
                    if self.cash_policy == "convert_at_fill" and div_ccy != base:
                        convert(amount, div_ccy, base, f"dividend {iid}")

        # 2. fills
        for order in [o for o in st.orders.values() if o.is_open]:
            iid = order.instrument_id
            inst = self.instruments[iid]
            if order.eligible_from is None or t < order.eligible_from:
                continue
            bar = bars.get(iid)
            cal_open = get_calendar(inst.calendar_code).is_session(t)
            if bar is None or not math.isfinite(bar.close):
                if cal_open and order.tif is TimeInForce.DAY:
                    order.close(OrderStatus.EXPIRED, "no bar on the eligible session")
                continue
            if order.tif is TimeInForce.DAY and t > order.eligible_from:
                order.close(OrderStatus.EXPIRED, "day order outlived its session (session not processed)")
                continue
            decision = fill_price(order, bar)  # type: ignore[arg-type]
            if decision.triggered:
                order.triggered = True
            if decision.rest_as_limit:
                order.order_type = OrderType.LIMIT
                order.notes.append("stop triggered; resting as limit")
            if decision.price is None:
                if order.tif is TimeInForce.DAY:
                    order.close(OrderStatus.EXPIRED, "not filled on its day")
                continue
            price = min(max(decision.price, bar.low), bar.high)
            qty = max_fillable(order.remaining, bar.volume, self.participation_cap)
            qty = round_quantity(qty, inst.lot_size, self.fractional_shares)
            participation = qty / bar.volume if bar.volume and bar.volume > 0 else None
            local = inst.currency.value
            if order.side is Side.BUY:
                costs = self.cost_model.fill_costs(inst, Side.BUY, qty, price, t, participation)
                need = qty * price + costs.total
                if self.cash_policy == "convert_at_fill" and local != base and st.cash.get(local, 0.0) < need:
                    shortfall = need - st.cash.get(local, 0.0)
                    rate = self._fx(base, local, t)
                    gross_base = shortfall / rate / (1 - self.cost_model.fx_cost_bps / 1e4)
                    convert(min(gross_base, max(st.cash.get(base, 0.0), 0.0)), base, local, f"fund buy {iid}")
                available = st.cash.get(local, 0.0)
                if available < need and qty > 0:
                    unit = price * (1 + (costs.total / (qty * price) if qty * price else 0))
                    affordable = round_quantity(
                        max(available, 0.0) / unit, inst.lot_size, self.fractional_shares
                    )
                    if affordable < qty:
                        order.notes.append(f"cash-constrained: {qty:g} -> {affordable:g}")
                        qty = affordable
                        costs = (
                            self.cost_model.fill_costs(inst, Side.BUY, qty, price, t, participation)
                            if qty
                            else costs
                        )
            else:
                if not self.allow_short:
                    held = st.positions.get(iid, PositionState(0.0, 0.0, local)).qty
                    if qty > held + 1e-12:
                        order.notes.append(f"long-only: {qty:g} -> {held:g}")
                        qty = round_quantity(held, inst.lot_size, self.fractional_shares)
                costs = self.cost_model.fill_costs(inst, Side.SELL, qty, price, t, participation)
            if qty <= 0:
                if order.tif is TimeInForce.DAY:
                    order.close(OrderStatus.EXPIRED, "zero fillable quantity")
                continue
            signed = qty if order.side is Side.BUY else -qty
            pos = st.positions.setdefault(iid, PositionState(0.0, 0.0, local))
            if signed > 0 and pos.qty >= 0:
                pos.avg_price = (pos.avg_price * pos.qty + qty * price) / (pos.qty + qty)
            elif signed < 0 and pos.qty <= 0:
                pos.avg_price = (pos.avg_price * abs(pos.qty) + qty * price) / (abs(pos.qty) + qty)
            pos.qty += signed
            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
                pos.avg_price = 0.0
            st.cash[local] = st.cash.get(local, 0.0) - signed * price
            st.add_event(
                "fill", t, f"{order.order_id} {order.side.value} {qty:g} @ {price:.6g}", instrument_id=iid,
                amount=-signed * price, currency=local,
            )  # fmt: skip
            if costs.total:
                st.cash[local] -= costs.total
                st.add_event(
                    "cost",
                    t,
                    f"{order.order_id} {costs.as_dict()}",
                    instrument_id=iid,
                    amount=-costs.total,
                    currency=local,
                )
            fill = BrokerFill(
                st.next_fill_id(),
                order.order_id,
                iid,
                order.side,
                qty,
                price,
                t,
                local,
                costs.commission,
                costs.spread,
                costs.slippage,
                costs.tax_total,
                0.0,
                order.tag,
            )
            st.fills.append(fill)
            new_fills.append(fill)
            order.record_fill(qty, price)
            turnover_base += qty * price * self._fx(local, base, t)
            if order.side is Side.SELL and self.cash_policy == "convert_at_fill" and local != base:
                if st.cash.get(local, 0.0) > 0:
                    convert(st.cash[local], local, base, f"sweep proceeds {iid}")
            if order.is_open and order.tif is TimeInForce.DAY:
                order.close(OrderStatus.EXPIRED, "partial fill; remainder expired")

        # 3. borrow on shorts at yesterday's marks
        for iid, pos in st.positions.items():
            if pos.qty < 0 and iid in st.marks:
                notional = abs(pos.qty) * st.marks[iid] * self._fx(pos.currency, base, t)
                cost = self.cost_model.daily_borrow_cost(notional)
                if cost:
                    st.cash[base] = st.cash.get(base, 0.0) - cost
                    st.add_event("borrow", t, "daily borrow", instrument_id=iid, amount=-cost, currency=base)

        # 4. marks and equity
        for iid, bar in bars.items():
            if iid in self.instruments and math.isfinite(bar.close):
                st.marks[iid] = bar.close
        st.as_of = t
        cash_base = self._cash_base(t)
        pv, gross, net = self._positions_value(t)
        equity = cash_base + pv
        st.equity.append(
            {
                "session_date": t,
                "equity": equity,
                "cash_base": cash_base,
                "positions_value": pv,
                "gross_exposure": gross / equity if equity else 0.0,
                "net_exposure": net / equity if equity else 0.0,
                "turnover": turnover_base / equity if equity else 0.0,
                "n_positions": sum(1 for p in st.positions.values() if abs(p.qty) > 1e-12),
                "fills": len(new_fills),
            }
        )
        self._persist()
        return new_fills
