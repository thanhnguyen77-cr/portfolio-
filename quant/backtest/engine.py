"""Event-driven multi-currency backtest engine.

Per union-calendar session t, in this order:
  1. corporate actions with ex-date t, on positions held at the previous close (splits scale quantity for
     RAW-priced series; cash dividends are credited in their own currency);
  2. fills of eligible orders against bar t (next-open semantics; partial fills via participation cap);
  3. daily borrow accrual on short positions;
  4. mark-to-market at the close of t in the base currency (closed markets carry their last close);
  5. the strategy decides with data <= t, adjusted as of t; new orders become eligible from the next session
     of their own instrument's calendar.

Look-ahead is prevented by construction: the strategy's Context can only reach bars up to t and the fill
step never sees a bar before the order exists.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from quant.backtest.costs import CostModel
from quant.backtest.fills import BarView, fill_price, max_fillable, round_quantity
from quant.backtest.metrics import compute_metrics
from quant.backtest.orders import Fill, Order, OrderStatus, OrderType, TimeInForce
from quant.backtest.result import BacktestResult
from quant.backtest.strategy import Context, Strategy
from quant.core.calendars import get_calendar, union_sessions
from quant.core.errors import BacktestError
from quant.core.types import AdjustmentState, Currency, Instrument, Side
from quant.data.corporate_actions import adjust_bars
from quant.data.fx import FIXING_NOTE, FXRates

_VARIANTS: dict[str, int] = defaultdict(int)


def variant_count(strategy_name: str) -> int:
    return _VARIANTS[strategy_name]


def reset_variant_counter() -> None:
    _VARIANTS.clear()


class BacktestConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: date
    end: date
    initial_cash: float = Field(default=1_000_000.0, gt=0)
    base_currency: Currency = Currency.USD
    cost_model: CostModel = Field(default_factory=CostModel)
    participation_cap: float | None = Field(default=0.10, description="Max fraction of bar volume per fill.")
    allow_short: bool = False
    cash_policy: Literal["convert_at_fill", "hold_foreign"] = "convert_at_fill"
    fractional_shares: bool = False
    min_notional: float = Field(
        default=100.0, ge=0, description="Skip rebalancing trades below this (base ccy)."
    )
    max_gross_leverage: float = Field(default=1.0, gt=0)
    adjust_as_of: bool = True
    decide_on_last_session: bool = False

    def describe(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"cost_model"})
        d["cost_model"] = self.cost_model.describe()
        return d


class Backtest:
    def __init__(
        self,
        config: BacktestConfig,
        strategy: Strategy,
        bars: pd.DataFrame,
        instruments: Mapping[str, Instrument],
        actions: pd.DataFrame | None = None,
        fx: FXRates | None = None,
        benchmark: pd.Series | None = None,
        caveats: Sequence[str] = (),
    ) -> None:
        self.cfg = config
        self.strategy = strategy
        missing = [iid for iid in strategy.instruments if iid not in instruments]
        if missing:
            raise BacktestError(f"strategy references unknown instruments {missing}")
        self.instruments = {iid: instruments[iid] for iid in strategy.instruments}
        self.actions = (
            actions
            if actions is not None
            else pd.DataFrame(columns=["instrument_id", "ex_date", "action_type", "value", "currency"])
        )
        self.fx = fx
        self.benchmark = benchmark
        self.extra_caveats = list(caveats)
        self._bars: dict[str, pd.DataFrame] = {}
        self._rows: dict[str, dict[pd.Timestamp, int]] = {}
        self._state: dict[str, str] = {}
        for iid, g in bars[bars["instrument_id"].isin(self.instruments)].groupby("instrument_id"):
            g = g.sort_values("session_date").reset_index(drop=True)
            self._bars[str(iid)] = g
            self._rows[str(iid)] = {ts: i for i, ts in enumerate(g["session_date"])}
            self._state[str(iid)] = str(g["adjustment_state"].iloc[0])
        for iid in self.instruments:
            if iid not in self._bars:
                raise BacktestError(f"no bars for {iid} in the supplied data")
        self._hist_cache: dict[tuple[str, int], pd.DataFrame] = {}

    # ------------------------------------------------------------------ helpers
    def _fx(self, from_ccy: str, to_ccy: str, on: date) -> float:
        if from_ccy == to_ccy:
            return 1.0
        if self.fx is None:
            raise BacktestError(f"FX rates are required to convert {from_ccy}->{to_ccy}")
        return self.fx.latest(from_ccy, to_ccy, on)

    def _bar(self, iid: str, on: pd.Timestamp) -> BarView | None:
        idx = self._rows[iid].get(on)
        if idx is None:
            return None
        r = self._bars[iid].iloc[idx]
        if not math.isfinite(r["close"]):
            return None
        return BarView(
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"])
        )

    def _history(self, iid: str, on: pd.Timestamp, n: int | None) -> pd.DataFrame:
        g = self._bars[iid]
        upto = g[g["session_date"] <= on]
        acts = (
            self.actions[(self.actions["instrument_id"] == iid)] if not self.actions.empty else self.actions
        )
        if self.cfg.adjust_as_of and not acts.empty:
            n_acts = int((pd.to_datetime(acts["ex_date"]) <= on).sum())
        else:
            n_acts = 0
        key = (iid, n_acts)
        cached = self._hist_cache.get(key)
        if cached is None or len(cached) < len(upto):
            cached = (
                adjust_bars(upto, acts, as_of=on.date())
                if self.cfg.adjust_as_of
                else adjust_bars(upto, acts.iloc[0:0])
            )
            self._hist_cache[key] = cached
        out = cached[cached["session_date"] <= on]
        return out.tail(n) if n else out

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        cfg = self.cfg
        base = cfg.base_currency.value
        codes = sorted({inst.calendar_code for inst in self.instruments.values()})
        sessions = union_sessions(codes, cfg.start, cfg.end)
        if len(sessions) == 0:
            raise BacktestError("no sessions in the requested window")
        _VARIANTS[self.strategy.name] += 1

        cash: dict[str, float] = defaultdict(float)
        cash[base] = float(cfg.initial_cash)
        positions: dict[str, float] = defaultdict(float)
        last_close: dict[str, float] = {}
        open_orders: list[Order] = []
        all_orders: list[Order] = []
        fills: list[Fill] = []
        events: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        cost_totals: dict[str, float] = defaultdict(float)
        next_id = 1
        self.strategy.on_start(self.instruments)
        acts_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
        if not self.actions.empty:
            a = self.actions.copy()
            a["ex_date"] = pd.to_datetime(a["ex_date"]).dt.normalize()
            acts_by_date = {ts: g for ts, g in a.groupby("ex_date")}

        def to_base(amount: float, ccy: str, on: date) -> float:
            return amount * self._fx(ccy, base, on)

        def convert(amount_from: float, from_ccy: str, to_ccy: str, on: date, reason: str) -> float:
            """Convert from_ccy -> to_ccy at today's fix, paying the FX cost; returns the amount credited."""
            if from_ccy == to_ccy or amount_from <= 0:
                return amount_from
            rate = self._fx(from_ccy, to_ccy, on)
            fee_from = cfg.cost_model.fx_conversion_cost(amount_from)
            credited = (amount_from - fee_from) * rate
            cash[from_ccy] -= amount_from
            cash[to_ccy] += credited
            cost_totals["fx_cost"] += to_base(fee_from, from_ccy, on)
            events.append(
                {
                    "session_date": on,
                    "type": "fx_conversion",
                    "instrument_id": None,
                    "detail": reason,
                    "amount": amount_from,
                    "currency": from_ccy,
                    "credited": credited,
                    "to_currency": to_ccy,
                }
            )
            return credited

        for s_idx, ts in enumerate(sessions):
            t: date = ts.date()

            # 1. corporate actions on positions held at the previous close
            todays_acts = acts_by_date.get(ts)
            if todays_acts is not None:
                for iid, kind, value, ccy in todays_acts[
                    ["instrument_id", "action_type", "value", "currency"]
                ].itertuples(index=False):
                    qty = positions.get(iid, 0.0)
                    if not qty or iid not in self.instruments:
                        continue
                    if kind == "split" and value > 0 and self._state.get(iid) == AdjustmentState.RAW.value:
                        positions[iid] = qty * float(value)
                        events.append(
                            {
                                "session_date": t,
                                "type": "split",
                                "instrument_id": iid,
                                "detail": f"ratio {value}",
                                "amount": qty * float(value) - qty,
                                "currency": None,
                            }
                        )
                    elif kind == "dividend" and value > 0:
                        div_ccy = ccy if isinstance(ccy, str) else self.instruments[iid].currency.value
                        amount = positions[iid] * float(value)
                        cash[div_ccy] += amount
                        events.append(
                            {
                                "session_date": t,
                                "type": "dividend",
                                "instrument_id": iid,
                                "detail": f"{value} per share",
                                "amount": amount,
                                "currency": div_ccy,
                            }
                        )
                        if cfg.cash_policy == "convert_at_fill" and div_ccy != base:
                            convert(amount, div_ccy, base, t, f"dividend {iid}")

            # 2. fills
            session_turnover_base = 0.0
            for order in list(open_orders):
                iid = order.instrument_id
                inst = self.instruments[iid]
                if order.eligible_from is None or t < order.eligible_from:
                    continue
                bar = self._bar(iid, ts)
                cal_open = get_calendar(inst.calendar_code).is_session(t)
                if bar is None:
                    if cal_open and order.tif is TimeInForce.DAY:
                        order.close(OrderStatus.EXPIRED, "no bar on the eligible session")
                        open_orders.remove(order)
                    continue
                decision = fill_price(order, bar)
                if decision.triggered:
                    order.triggered = True
                if decision.rest_as_limit:
                    order.order_type = OrderType.LIMIT
                    order.notes.append("stop triggered; resting as limit")
                if decision.price is None:
                    if order.tif is TimeInForce.DAY:
                        order.close(OrderStatus.EXPIRED, "not filled on its day")
                        open_orders.remove(order)
                    continue
                price = min(max(decision.price, bar.low), bar.high)
                qty = max_fillable(order.remaining, bar.volume, cfg.participation_cap)
                qty = round_quantity(qty, inst.lot_size, cfg.fractional_shares)
                participation = qty / bar.volume if bar.volume and bar.volume > 0 else None
                local = inst.currency.value
                if order.side is Side.BUY:
                    costs = cfg.cost_model.fill_costs(inst, Side.BUY, qty, price, t, participation)
                    need = qty * price + costs.total
                    if cfg.cash_policy == "convert_at_fill" and local != base and cash[local] < need:
                        shortfall = need - cash[local]
                        rate = self._fx(base, local, t)
                        gross_base = shortfall / rate / (1 - cfg.cost_model.fx_cost_bps / 1e4)
                        convert(min(gross_base, max(cash[base], 0.0)), base, local, t, f"fund buy {iid}")
                    available = cash[local]
                    if available < need and qty > 0:
                        unit = price * (1 + (costs.total / (qty * price) if qty * price else 0))
                        affordable = round_quantity(
                            max(available, 0.0) / unit, inst.lot_size, cfg.fractional_shares
                        )
                        if affordable < qty:
                            order.notes.append(f"cash-constrained: {qty:g} -> {affordable:g}")
                            qty = affordable
                            costs = (
                                cfg.cost_model.fill_costs(inst, Side.BUY, qty, price, t, participation)
                                if qty
                                else costs
                            )
                else:
                    if not cfg.allow_short:
                        held = positions.get(iid, 0.0)
                        if qty > held + 1e-12:
                            order.notes.append(f"long-only: {qty:g} -> {held:g}")
                            qty = round_quantity(held, inst.lot_size, cfg.fractional_shares)
                    costs = cfg.cost_model.fill_costs(inst, Side.SELL, qty, price, t, participation)
                if qty <= 0:
                    if order.tif is TimeInForce.DAY:
                        order.close(OrderStatus.EXPIRED, "zero fillable quantity")
                        open_orders.remove(order)
                    continue
                signed = qty if order.side is Side.BUY else -qty
                positions[iid] += signed
                cash[local] -= signed * price
                cash[local] -= costs.total
                for name, amount in (
                    ("commission", costs.commission),
                    ("spread", costs.spread),
                    ("slippage", costs.slippage),
                ):
                    cost_totals[name] += to_base(amount, local, t)
                for name, amount in (costs.taxes or {}).items():
                    cost_totals[name] += to_base(amount, local, t)
                fill = Fill(
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
                fills.append(fill)
                order.record_fill(qty, price)
                session_turnover_base += to_base(qty * price, local, t)
                if (
                    order.side is Side.SELL
                    and cfg.cash_policy == "convert_at_fill"
                    and local != base
                    and cash[local] > 0
                ):
                    convert(cash[local], local, base, t, f"sweep proceeds {iid}")
                if not order.is_open:
                    open_orders.remove(order)
                elif order.tif is TimeInForce.DAY:
                    order.close(OrderStatus.EXPIRED, "partial fill; remainder expired")
                    open_orders.remove(order)
                if abs(positions[iid]) < 1e-12:
                    positions[iid] = 0.0

            # 3. borrow on shorts (yesterday's marks)
            for iid, qty in positions.items():
                if qty < 0 and iid in last_close:
                    notional = (
                        abs(qty) * last_close[iid] * self._fx(self.instruments[iid].currency.value, base, t)
                    )
                    cost = cfg.cost_model.daily_borrow_cost(notional)
                    cash[base] -= cost
                    cost_totals["borrow"] += cost
                    events.append(
                        {
                            "session_date": t,
                            "type": "borrow",
                            "instrument_id": iid,
                            "detail": "daily borrow",
                            "amount": -cost,
                            "currency": base,
                        }
                    )

            # 4. mark to market
            for iid in self.instruments:
                bar = self._bar(iid, ts)
                if bar is not None:
                    last_close[iid] = bar.close
            positions_value = gross = net = 0.0
            for iid, qty in positions.items():
                if qty and iid in last_close:
                    v = qty * last_close[iid] * self._fx(self.instruments[iid].currency.value, base, t)
                    positions_value += v
                    gross += abs(v)
                    net += v
                    position_rows.append(
                        {
                            "session_date": t,
                            "instrument_id": iid,
                            "qty": qty,
                            "price_local": last_close[iid],
                            "value_base": v,
                        }
                    )
            cash_base = sum(to_base(amt, ccy, t) for ccy, amt in cash.items() if amt)
            equity = cash_base + positions_value
            equity_rows.append(
                {
                    "session_date": t,
                    "equity": equity,
                    "cash_base": cash_base,
                    "positions_value": positions_value,
                    "gross_exposure": gross / equity if equity else 0.0,
                    "net_exposure": net / equity if equity else 0.0,
                    "turnover": session_turnover_base / equity if equity else 0.0,
                    "n_positions": sum(1 for q in positions.values() if q),
                    "fills": sum(1 for f in fills if f.session_date == t),
                }
            )

            # 5. decide
            is_last = s_idx == len(sessions) - 1
            if (not is_last or cfg.decide_on_last_session) and s_idx + 1 >= self.strategy.warmup:
                ctx = Context(
                    session_date=t,
                    instruments=self.instruments,
                    base_currency=base,
                    positions=dict(positions),
                    cash=dict(cash),
                    equity=equity,
                    allow_short=cfg.allow_short,
                    fractional_shares=cfg.fractional_shares,
                    min_notional=cfg.min_notional,
                    _history=lambda iid, n, _ts=ts: self._history(iid, _ts, n),
                    _close=lambda iid: last_close.get(iid),
                    _fx=lambda a, b, _t=t: self._fx(a, b, _t),
                    session_index=s_idx,
                )
                decision = self.strategy.on_session(ctx)
                new_orders: list[Order] = []
                if isinstance(decision, Mapping):
                    new_orders = ctx.target_weights(decision, tag=self.strategy.name)
                elif decision:
                    new_orders = list(decision)
                if new_orders and cfg.max_gross_leverage:
                    new_orders = self._apply_leverage_cap(new_orders, positions, last_close, equity, t)
                for order in new_orders:
                    if order.instrument_id not in self.instruments:
                        raise BacktestError(f"order for unknown instrument {order.instrument_id!r}")
                    order.order_id = next_id
                    next_id += 1
                    order.created_session = t
                    cal = get_calendar(self.instruments[order.instrument_id].calendar_code)
                    order.eligible_from = cal.next_session(t)
                    open_orders.append(order)
                    all_orders.append(order)

        for order in open_orders:
            order.close(OrderStatus.CANCELLED, "end of backtest")

        equity_df = pd.DataFrame(equity_rows)
        equity_df["session_date"] = pd.to_datetime(equity_df["session_date"])
        eq_series = pd.Series(
            equity_df["equity"].to_numpy(), index=pd.DatetimeIndex(equity_df["session_date"])
        )
        metrics = compute_metrics(
            eq_series,
            self.benchmark,
            turnover=pd.Series(equity_df["turnover"].to_numpy(), index=eq_series.index),
            gross_exposure=pd.Series(equity_df["gross_exposure"].to_numpy(), index=eq_series.index),
        )
        fills_df = (
            pd.DataFrame([f.as_dict() for f in fills])
            if fills
            else pd.DataFrame(columns=list(Fill(None, "", Side.BUY, 1, 1, cfg.start, base).as_dict()))
        )
        orders_df = (
            pd.DataFrame([o.as_dict() for o in all_orders])
            if all_orders
            else pd.DataFrame(columns=list(Order("x", Side.BUY, 1).as_dict()))
        )
        events_df = (
            pd.DataFrame(events)
            if events
            else pd.DataFrame(
                columns=["session_date", "type", "instrument_id", "detail", "amount", "currency"]
            )
        )
        positions_df = (
            pd.DataFrame(position_rows)
            if position_rows
            else pd.DataFrame(columns=["session_date", "instrument_id", "qty", "price_local", "value_base"])
        )
        return BacktestResult(
            strategy=self.strategy.describe(),
            config=cfg.describe(),
            equity=equity_df,
            fills=fills_df,
            orders=orders_df,
            positions=positions_df,
            events=events_df,
            metrics=metrics,
            costs={k: float(v) for k, v in sorted(cost_totals.items()) if v},
            caveats=self._caveats(),
            variant_count=_VARIANTS[self.strategy.name],
            benchmark=self.benchmark,
        )

    def _apply_leverage_cap(
        self,
        orders: list[Order],
        positions: Mapping[str, float],
        last_close: Mapping[str, float],
        equity: float,
        on: date,
    ) -> list[Order]:
        """Scale down new market orders so projected gross exposure stays within max_gross_leverage."""
        if equity <= 0:
            return []
        base = self.cfg.base_currency.value
        projected = dict(positions)
        for o in orders:
            projected[o.instrument_id] = projected.get(o.instrument_id, 0.0) + o.signed_qty
        gross = 0.0
        for iid, qty in projected.items():
            px = last_close.get(iid)
            if qty and px:
                gross += abs(qty) * px * self._fx(self.instruments[iid].currency.value, base, on)
        limit = self.cfg.max_gross_leverage * equity
        if gross <= limit * (1 + 1e-9):
            return orders
        scale = limit / gross
        kept: list[Order] = []
        for o in orders:
            inst = self.instruments[o.instrument_id]
            is_reducing = (positions.get(o.instrument_id, 0.0) > 0 and o.side is Side.SELL) or (
                positions.get(o.instrument_id, 0.0) < 0 and o.side is Side.BUY
            )
            if is_reducing:
                kept.append(o)
                continue
            new_qty = round_quantity(o.qty * scale, inst.lot_size, self.cfg.fractional_shares)
            if new_qty > 0:
                o.qty = new_qty
                o.notes.append(f"scaled for gross leverage cap {self.cfg.max_gross_leverage:g}")
                kept.append(o)
        return kept

    def _caveats(self) -> list[str]:
        cfg = self.cfg
        out = [
            f"Fills at the next session open of each market; participation cap {cfg.participation_cap}; "
            "partial fills allowed; day orders expire unfilled.",
            "Costs: "
            + ", ".join(
                f"{k}={v}"
                for k, v in cfg.cost_model.model_dump(
                    exclude={"fee_schedule", "half_spread_bps_by_instrument"}
                ).items()
            ),
            "Dividends are credited on the ex-date (pay-date lag ignored); "
            "splits scale quantity only for RAW-priced series.",
            "Markets closed on a union-calendar session are marked at their last available close.",
        ]
        states = sorted(set(self._state.values()))
        out.append(
            f"Vendor adjustment state(s): {states}; "
            "signals use as-of adjusted prices, fills use stored prices."
        )
        unverified = cfg.cost_model.describe().get("unverified_fee_rules", [])
        if unverified:
            out.append(f"Fee-schedule rules not yet verified against primary sources: {unverified}.")
        ccys = {inst.currency.value for inst in self.instruments.values()}
        if ccys - {cfg.base_currency.value}:
            out.append(FIXING_NOTE)
        out.append(
            "Universe/survivorship quality was not supplied to this run; treat results as an upper bound."
        )
        out += self.extra_caveats
        out.append(
            f"{_VARIANTS[self.strategy.name]} variant(s) of '{self.strategy.name}' were run in this process; "
            "the deflated Sharpe ratio (stage S5) accounts for this."
        )
        return out
