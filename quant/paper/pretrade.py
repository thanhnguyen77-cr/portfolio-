"""Pre-trade gate: every order the runner (or a human) sends to a paper broker passes these checks first.

The rule list is ported from the trader plugin's risk_check.py and generalised to instruments, currencies
and the broker snapshot types. Each check is pass / warn / fail with the evidence; any fail blocks the
order. "Paper" is the only mode that passes the `mode` check: a live request fails here before it can
reach a broker, independently of the live-trading gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, Field

from quant.backtest.orders import OrderType
from quant.core.types import Instrument, Side
from quant.paper.broker import AccountSnapshot, BrokerOrder, OrderRequest, Position


class PreTradeRules(BaseModel):
    allow_shorts: bool = False
    allow_market_orders: bool = True
    min_price: float = Field(default=1.0, ge=0, description="Reference price floor (local currency).")
    max_position_pct: float = Field(
        default=10.0, gt=0, description="Max value in one instrument, % of equity."
    )
    max_order_pct: float | None = Field(default=None, description="Max notional of one order, % of equity.")
    max_risk_per_trade_pct: float = Field(
        default=1.0, gt=0, description="Max (entry - stop) * qty, % equity."
    )
    require_protective_stop: bool = False
    min_reward_risk: float = Field(default=1.5, ge=0)
    cash_reserve_pct: float = Field(default=5.0, ge=0, description="Cash to keep after buys, % of equity.")
    max_total_exposure_pct: float = Field(default=100.0, gt=0)
    max_open_positions: int = Field(default=20, ge=1)
    max_daily_loss_pct: float = Field(default=3.0, gt=0)
    blocklist: set[str] = Field(default_factory=set)
    earnings_blackout_days: int = Field(default=2, ge=0)
    blocked_regimes: set[str] = Field(default_factory=lambda: {"risk_off"})
    caution_regimes: set[str] = Field(default_factory=lambda: {"caution"})
    max_participation: float | None = Field(default=0.10, description="Order qty / ADV ceiling.")
    allow_duplicate_orders: bool = False
    fractional_shares: bool = False


@dataclass
class Check:
    name: str
    status: str  # pass | warn | fail
    detail: str


@dataclass
class PreTradeReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    def ok(self, name: str, condition: bool, detail: str) -> None:
        self.add(name, "pass" if condition else "fail", detail)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([c.__dict__ for c in self.checks], columns=["name", "status", "detail"])

    def summary(self) -> str:
        verdict = "PASS" if self.passed else "BLOCK"
        parts = [
            f"{verdict}: {len(self.failures)} fail, {len(self.warnings)} warn, {len(self.checks)} checks"
        ]
        parts += [f"  - {c.name}: {c.detail}" for c in self.failures]
        return "\n".join(parts)


def check_order(
    request: OrderRequest,
    rules: PreTradeRules,
    account: AccountSnapshot,
    positions: Sequence[Position],
    open_orders: Sequence[BrokerOrder],
    *,
    instrument: Instrument | None,
    ref_price: float | None,
    fx_rate: float = 1.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    adv: float | None = None,
    regime: str | None = None,
    days_to_earnings: int | None = None,
    live: bool = False,
) -> PreTradeReport:
    """`ref_price`, `stop_loss`, `take_profit` in the instrument's currency; `fx_rate` converts to base."""
    ck = PreTradeReport()
    iid = request.instrument_id
    qty = float(request.qty)
    side = request.side
    equity = float(account.equity)

    # --- mode -----------------------------------------------------------------------------------
    if live:
        ck.add("mode", "fail", "live orders are not supported: this platform is paper-trading only")
    else:
        ck.add("mode", "pass", f"paper trading ({account.broker})")

    # --- instrument and order shape -----------------------------------------------------------
    if instrument is None:
        ck.add("instrument", "fail", f"unknown instrument {iid!r}")
        return ck
    ck.ok("instrument", bool(instrument.tradable), f"{iid} tradable={instrument.tradable}")
    ck.ok(
        "blocklist",
        iid not in rules.blocklist and instrument.symbol not in rules.blocklist,
        f"{iid} not blocklisted",
    )
    held = next((p for p in positions if p.instrument_id == iid), None)
    held_qty = held.qty if held else 0.0
    if side is Side.BUY:
        ck.add("side", "pass", "buy")
    elif rules.allow_shorts:
        ck.add("side", "warn", "sell with shorts allowed by the rules")
    else:
        ck.ok("side", qty <= held_qty + 1e-9, f"sell {qty:g} vs held {held_qty:g} (shorts disabled)")
    whole = (
        rules.fractional_shares or abs(qty / instrument.lot_size - round(qty / instrument.lot_size)) < 1e-9
    )
    ck.ok(
        "qty",
        qty > 0 and whole,
        f"qty {qty:g} (lot {instrument.lot_size}, fractional={rules.fractional_shares})",
    )
    allowed = {OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT} | (
        {OrderType.MARKET} if rules.allow_market_orders else set()
    )
    ck.ok(
        "order_type",
        request.order_type in allowed,
        f"{request.order_type.value} (allowed: {', '.join(sorted(t.value for t in allowed))})",
    )
    if request.order_type is OrderType.STOP_LIMIT:
        lim, stp = float(request.limit_price or 0), float(request.stop_price or 0)
        consistent = lim >= stp if side is Side.BUY else lim <= stp
        ck.ok("stop_limit_prices", consistent, f"stop {stp}, limit {lim}")
    worst = request.limit_price or request.stop_price or ref_price  # worst plausible fill for cost/risk
    planned = (
        request.stop_price if (request.order_type is OrderType.STOP_LIMIT and request.stop_price) else worst
    )
    ck.ok("entry_price", bool(worst and worst > 0), f"reference price {worst}")
    if not worst or not worst > 0:
        return ck
    ck.ok("min_price", worst >= rules.min_price, f"{worst} vs min {rules.min_price}")

    # --- protection and reward:risk (buys) ----------------------------------------------------
    if side is Side.BUY:
        if stop_loss is not None or rules.require_protective_stop:
            ck.ok(
                "protective_stop",
                bool(stop_loss and stop_loss < planned),
                f"stop {stop_loss} below entry {planned}",
            )
        if take_profit is not None:
            ck.ok("take_profit", take_profit > worst, f"target {take_profit} above worst fill {worst}")
        if stop_loss and take_profit and stop_loss < planned < take_profit:
            rr = (take_profit - planned) / (planned - stop_loss)
            ck.ok(
                "reward_risk",
                rr >= rules.min_reward_risk - 1e-9,
                f"{rr:.2f} R vs min {rules.min_reward_risk}",
            )

    # --- money limits -------------------------------------------------------------------------
    cost_base = qty * worst * fx_rate
    existing_base = abs(held.market_value_base or 0.0) if held else 0.0
    if side is Side.BUY:
        cap = equity * rules.max_position_pct / 100.0
        ck.ok(
            "max_position_pct",
            cost_base + existing_base <= cap + 1e-6,
            f"{cost_base + existing_base:,.2f} in {iid} vs max {cap:,.2f} "
            f"({rules.max_position_pct}% of equity)",
        )
        if rules.max_order_pct is not None:
            ocap = equity * rules.max_order_pct / 100.0
            ck.ok("max_order_pct", cost_base <= ocap + 1e-6, f"order {cost_base:,.2f} vs max {ocap:,.2f}")
        if stop_loss and stop_loss < worst:
            risk_base = qty * (worst - stop_loss) * fx_rate
            rcap = equity * rules.max_risk_per_trade_pct / 100.0
            ck.ok(
                "max_risk_per_trade",
                risk_base <= rcap * 1.001,
                f"{risk_base:,.2f} at risk vs max {rcap:,.2f}",
            )
        pending = sum(
            (o.remaining * (o.limit_price or o.stop_price or 0.0)) * fx_rate
            for o in open_orders
            if o.side is Side.BUY
        )
        cash_after = account.cash_base - pending - cost_base
        reserve = equity * rules.cash_reserve_pct / 100.0
        ck.ok(
            "cash_reserve",
            cash_after >= reserve - 1e-6,
            f"cash after {cash_after:,.2f} vs reserve {reserve:,.2f}",
        )
        gross = sum(abs(p.market_value_base or 0.0) for p in positions)
        ecap = equity * rules.max_total_exposure_pct / 100.0
        ck.ok(
            "total_exposure",
            gross + pending + cost_base <= ecap + 1e-6,
            f"exposure after {gross + pending + cost_base:,.2f} vs cap {ecap:,.2f}",
        )
        n_after = len({p.instrument_id for p in positions if abs(p.qty) > 1e-12} | {iid})
        ck.ok(
            "max_open_positions",
            n_after <= rules.max_open_positions,
            f"{n_after} vs max {rules.max_open_positions}",
        )
    dup = any(o.instrument_id == iid and o.side is side for o in open_orders)
    if rules.allow_duplicate_orders:
        ck.add("no_duplicate", "pass", "duplicates allowed by rules")
    else:
        ck.ok("no_duplicate", not dup, f"no open {side.value} order already resting in {iid}")
    if account.last_equity:
        day = (equity / account.last_equity - 1.0) * 100.0
        ck.ok(
            "daily_loss_limit",
            day > -rules.max_daily_loss_pct,
            f"today {day:+.2f}% vs limit -{rules.max_daily_loss_pct}%",
        )
    else:
        ck.add("daily_loss_limit", "warn", "no previous equity mark; daily loss not checked")
    if rules.max_participation is not None:
        if adv:
            ck.ok(
                "participation",
                qty / adv <= rules.max_participation,
                f"{qty / adv:.1%} of ADV vs max {rules.max_participation:.0%}",
            )
        else:
            ck.add("participation", "warn", "no ADV supplied; participation not checked")

    # --- context ------------------------------------------------------------------------------
    if regime is None:
        ck.add("regime", "warn", "no market-regime read supplied")
    elif regime in rules.blocked_regimes and side is Side.BUY:
        ck.add("regime", "fail", f"market regime {regime}: no new longs")
    elif regime in rules.caution_regimes:
        ck.add("regime", "warn", f"market regime {regime}: size should be reduced")
    else:
        ck.add("regime", "pass", regime)
    if days_to_earnings is not None and side is Side.BUY:
        if 0 <= days_to_earnings <= rules.earnings_blackout_days:
            ck.add("earnings", "fail", f"earnings in {days_to_earnings} day(s): binary event risk")
        else:
            ck.add("earnings", "pass", f"earnings in {days_to_earnings} day(s)")
    return ck
