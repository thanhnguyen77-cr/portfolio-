"""PaperRunner: the daily paper-trading job.

For a session t (after its close): refresh cached bars if allowed -> replay every unprocessed session up to t
through the broker (fills of resting orders, corporate actions, marks) -> hand the strategy a Context that
sees only data up to t -> pass each resulting order through the pre-trade gate -> submit the survivors
(eligible from the next session) -> reconcile the ledger -> report. The ordering (fills, mark, decide)
mirrors the backtest engine exactly, so a strategy behaves the same in paper as in a backtest.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from quant.backtest.fills import BarView
from quant.backtest.orders import Order
from quant.backtest.strategy import Context, Strategy
from quant.core.calendars import latest_complete_session, union_sessions
from quant.core.errors import BrokerError
from quant.core.time import to_date
from quant.core.types import Instrument
from quant.data.corporate_actions import adjust_bars
from quant.data.fx import FXRates
from quant.data.service import DataService
from quant.paper.broker import AccountSnapshot, Broker, BrokerFill, BrokerOrder, OrderRequest
from quant.paper.pretrade import PreTradeReport, PreTradeRules, check_order
from quant.paper.reconcile import ReconciliationReport, reconcile
from quant.paper.sim_broker import SimBroker

WARMUP_CALENDAR_DAYS = 420


@dataclass
class SessionReport:
    session_date: date
    processed_sessions: list[date]
    fills: list[BrokerFill]
    submitted: list[BrokerOrder]
    broker_rejected: list[BrokerOrder]
    blocked: list[tuple[OrderRequest, PreTradeReport]]
    account: AccountSnapshot
    reconciliation: ReconciliationReport | None
    notes: list[str] = field(default_factory=list)

    def blocked_table(self) -> pd.DataFrame:
        rows = []
        for req, rep in self.blocked:
            rows.append(
                {
                    **req.as_dict(),
                    "failures": "; ".join(f"{c.name}: {c.detail}" for c in rep.failures),
                    "warnings": "; ".join(c.name for c in rep.warnings),
                }
            )
        cols = list(OrderRequest("x", "buy", 1).as_dict()) + ["failures", "warnings"]
        return pd.DataFrame(rows, columns=cols)

    def summary(self) -> str:
        lines = [
            f"session {self.session_date}: processed {len(self.processed_sessions)} session(s), "
            f"{len(self.fills)} fill(s), {len(self.submitted)} order(s) submitted, "
            f"{len(self.blocked)} blocked by pre-trade rules, {len(self.broker_rejected)} rejected by broker",
            f"  equity {self.account.equity:,.2f} {self.account.base_currency}, "
            f"cash {self.account.cash_base:,.2f}, "
            f"{self.account.n_positions} position(s), {self.account.n_open_orders} open order(s)",
        ]
        if self.reconciliation is not None:
            lines.append("  reconciliation: " + self.reconciliation.summary().replace("\n", "\n  "))
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


class PaperRunner:
    def __init__(
        self,
        service: DataService,
        broker: Broker,
        strategy: Strategy,
        rules: PreTradeRules | None = None,
        *,
        refresh: bool | None = None,
        min_notional: float = 100.0,
        adjust_as_of: bool = True,
        regime_fn: Callable[[Context], str | None] | None = None,
        warmup_days: int = WARMUP_CALENDAR_DAYS,
    ) -> None:
        self.service = service
        self.broker = broker
        self.strategy = strategy
        self.rules = rules or PreTradeRules()
        self.refresh = refresh
        self.min_notional = min_notional
        self.adjust_as_of = adjust_as_of
        self.regime_fn = regime_fn
        self.warmup_days = warmup_days
        self.instruments: dict[str, Instrument] = {iid: service.resolve(iid) for iid in strategy.instruments}
        if isinstance(broker, SimBroker):
            for iid, inst in self.instruments.items():
                broker.instruments.setdefault(iid, inst)
        self.strategy.on_start(self.instruments)

    # ------------------------------------------------------------------ data
    @property
    def calendar_codes(self) -> list[str]:
        return sorted({i.calendar_code for i in self.instruments.values()})

    def default_session(self) -> date:
        return min(latest_complete_session(c) for c in self.calendar_codes)

    def load_data(self, end: date) -> tuple[pd.DataFrame, pd.DataFrame, FXRates | None]:
        ids = list(self.instruments)
        start = end - timedelta(days=self.warmup_days)
        bars = self.service.get_bars(ids, start, end, refresh=self.refresh)
        vendor = None if bars.empty else str(bars["vendor"].iloc[0])
        actions = self.service.db.load_corporate_actions(ids, vendor=vendor)
        base = (
            self.broker.account().base_currency
            if not isinstance(self.broker, SimBroker)
            else self.broker.base
        )
        foreign = sorted({i.currency.value for i in self.instruments.values()} - {base})
        fx = None
        if foreign:
            needed = {frozenset({base, f}) for f in foreign}
            pairs = [
                p
                for p in self.service.catalog.filter(asset_class="fx")
                if frozenset({p.symbol[:3], p.symbol[3:]}) in needed
            ]
            fx_bars = (
                self.service.get_bars(pairs, start, end, refresh=self.refresh) if pairs else bars.iloc[0:0]
            )
            fx = FXRates.from_bars(fx_bars, self.service.catalog)
        return bars, actions, fx

    @staticmethod
    def bar_views(bars: pd.DataFrame, session: date) -> dict[str, BarView]:
        day = bars[bars["session_date"] == pd.Timestamp(session)]
        out: dict[str, BarView] = {}
        for r in day.itertuples(index=False):
            if math.isfinite(float(r.close)):
                out[str(r.instrument_id)] = BarView(
                    float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)
                )
        return out

    # ------------------------------------------------------------------ steps
    def catch_up(
        self, bars: pd.DataFrame, actions: pd.DataFrame, until: date
    ) -> tuple[list[date], list[BrokerFill]]:
        """Replay unprocessed sessions (as_of, until] through a SimBroker; other brokers fill themselves."""
        if not isinstance(self.broker, SimBroker):
            return [], []
        st = self.broker.state
        if st.as_of is not None and until < st.as_of:
            raise BrokerError(f"session {until} is before the account's current session {st.as_of}")
        start = st.as_of + timedelta(days=1) if st.as_of is not None else until
        sessions = [s.date() for s in union_sessions(self.calendar_codes, start, until)]
        fills: list[BrokerFill] = []
        for s in sessions:
            fills += self.broker.process_session(s, self.bar_views(bars, s), actions)
        return sessions, fills

    def make_context(
        self, t: date, bars: pd.DataFrame, actions: pd.DataFrame, fx: FXRates | None, account: AccountSnapshot
    ) -> Context:
        ts = pd.Timestamp(t)
        by_id = {str(k): g.sort_values("session_date") for k, g in bars.groupby("instrument_id")}
        base = account.base_currency

        def history(iid: str, n: int | None) -> pd.DataFrame:
            g = by_id.get(iid)
            if g is None:
                return adjust_bars(bars.iloc[0:0], actions.iloc[0:0])
            upto = g[g["session_date"] <= ts]
            acts = actions[actions["instrument_id"] == iid] if not actions.empty else actions
            out = adjust_bars(upto, acts, as_of=t) if self.adjust_as_of else adjust_bars(upto, acts.iloc[0:0])
            return out.tail(n) if n else out

        def close(iid: str) -> float | None:
            g = by_id.get(iid)
            if g is None:
                return None
            upto = g[(g["session_date"] <= ts) & g["close"].notna()]
            return float(upto["close"].iloc[-1]) if len(upto) else None

        def fx_rate(a: str, b: str) -> float:
            if a == b:
                return 1.0
            if fx is None:
                raise BrokerError(f"FX rates required to convert {a}->{b}")
            return fx.latest(a, b, t)

        positions = {p.instrument_id: p.qty for p in self.broker.positions()}
        return Context(
            session_date=t,
            instruments=self.instruments,
            base_currency=base,
            positions=positions,
            cash=dict(account.cash),
            equity=account.equity,
            allow_short=self.rules.allow_shorts,
            fractional_shares=self.rules.fractional_shares,
            min_notional=self.min_notional,
            _history=history,
            _close=close,
            _fx=fx_rate,
            session_index=len(self.broker.state.equity) if isinstance(self.broker, SimBroker) else 0,
        )

    def run_session(self, session_date: date | str | None = None) -> SessionReport:
        t = to_date(session_date) if session_date is not None else self.default_session()
        notes: list[str] = []
        bars, actions, fx = self.load_data(t)
        if bars.empty:
            raise BrokerError("no bars available for the strategy's instruments; fetch data first")
        if isinstance(self.broker, SimBroker) and fx is not None:
            self.broker.fx = fx
        processed, fills = self.catch_up(bars, actions, t)
        if isinstance(self.broker, SimBroker) and self.broker.state.as_of != t:
            notes.append(f"no session data for {t}; broker state is as of {self.broker.state.as_of}")
        account = self.broker.account()
        ctx = self.make_context(t, bars, actions, fx, account)
        orders: list[Order] = []
        shortest = min((len(ctx.history(iid)) for iid in self.instruments), default=0)
        if self.strategy.warmup and shortest < self.strategy.warmup:
            notes.append(f"strategy warm-up not satisfied ({shortest} < {self.strategy.warmup} sessions)")
        else:
            decision = self.strategy.on_session(ctx)
            if isinstance(decision, Mapping):
                orders = ctx.target_weights(decision, tag=self.strategy.name)
            elif decision:
                orders = list(decision)
        submitted: list[BrokerOrder] = []
        rejected: list[BrokerOrder] = []
        blocked: list[tuple[OrderRequest, PreTradeReport]] = []
        regime = self.regime_fn(ctx) if self.regime_fn is not None else None
        for o in orders:
            req = OrderRequest(
                o.instrument_id,
                o.side,
                o.qty,
                o.order_type,
                limit_price=o.limit_price,
                stop_price=o.stop_price,
                tif=o.tif,
                tag=o.tag,
            )
            inst = self.instruments.get(o.instrument_id)
            ref = ctx.close(o.instrument_id)
            hist = ctx.history(o.instrument_id, 20)
            adv = float(hist["volume"].mean()) if len(hist) and hist["volume"].notna().any() else None
            report = check_order(
                req,
                self.rules,
                account,
                self.broker.positions(),
                self.broker.open_orders() + submitted,
                instrument=inst,
                ref_price=ref,
                fx_rate=ctx.fx(inst.currency.value, account.base_currency) if inst else 1.0,
                adv=adv,
                regime=regime,
            )
            if not report.passed:
                blocked.append((req, report))
                if isinstance(self.broker, SimBroker):
                    self.broker.state.add_event(
                        "pretrade_blocked", t, f"{req.side.value} {req.qty:g} {req.instrument_id}: "
                        + "; ".join(c.name for c in report.failures), instrument_id=req.instrument_id,
                    )  # fmt: skip
                continue
            bo = self.broker.submit(req)
            (submitted if bo.is_open or bo.filled_qty > 0 else rejected).append(bo)
        if isinstance(self.broker, SimBroker):
            self.broker._persist()
            recon: ReconciliationReport | None = reconcile(self.broker.state, self.broker)
        else:
            recon = None
        return SessionReport(
            session_date=t,
            processed_sessions=processed,
            fills=fills,
            submitted=submitted,
            broker_rejected=rejected,
            blocked=blocked,
            account=self.broker.account(),
            reconciliation=recon,
            notes=notes,
        )


def submit_manual(
    runner: PaperRunner,
    request: OrderRequest,
    *,
    session_date: date | str | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> tuple[PreTradeReport, BrokerOrder | None, list[BrokerFill]]:
    """Catch the broker up to `session_date`, gate one hand-written order, submit it if it passes."""
    t = to_date(session_date) if session_date is not None else runner.default_session()
    bars, actions, fx = runner.load_data(t)
    if isinstance(runner.broker, SimBroker) and fx is not None:
        runner.broker.fx = fx
    _, fills = runner.catch_up(bars, actions, t)
    account = runner.broker.account()
    ctx = runner.make_context(t, bars, actions, fx, account)
    inst = runner.instruments.get(request.instrument_id) or runner.service.resolve(request.instrument_id)
    if isinstance(runner.broker, SimBroker):
        runner.broker.instruments.setdefault(inst.instrument_id, inst)
    hist = ctx.history(request.instrument_id, 20)
    adv = float(hist["volume"].mean()) if len(hist) else None
    report = check_order(
        request,
        runner.rules,
        account,
        runner.broker.positions(),
        runner.broker.open_orders(),
        instrument=inst,
        ref_price=ctx.close(request.instrument_id),
        fx_rate=ctx.fx(inst.currency.value, account.base_currency),
        stop_loss=stop_loss,
        take_profit=take_profit,
        adv=adv,
    )
    if not report.passed:
        return report, None, fills
    return report, runner.broker.submit(request), fills


def runner_kwargs_from_settings(settings: Any) -> dict[str, Any]:
    return {"min_notional": 100.0, "adjust_as_of": True, "refresh": None if settings else False}
