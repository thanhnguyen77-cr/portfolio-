"""Persistent paper-trading ledger in DuckDB: accounts, per-currency cash, positions, orders, fills, marks,
equity history and an event log of every cash movement, so the book can be re-derived and checked.

Invariants the ledger can verify from its own rows:
  * positions == sum of signed fill quantities per instrument (adjusted for recorded splits);
  * cash per currency == initial deposit + sum of recorded cash events (fills, costs, dividends, FX legs,
    borrow). Any drift between the live state and these recomputations is a reconciliation issue.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from quant.core.errors import StoreError
from quant.core.time import now
from quant.data.store import DuckDBStore
from quant.paper.broker import BrokerFill, BrokerOrder

PAPER_TABLES = (
    "paper_events",
    "paper_equity",
    "paper_marks",
    "paper_fills",
    "paper_orders",
    "paper_positions",
    "paper_cash",
    "paper_accounts",
)


@dataclass
class PositionState:
    qty: float
    avg_price: float
    currency: str


@dataclass
class AccountState:
    account_id: str
    broker: str
    base_currency: str
    initial_cash: float
    created_at: datetime
    as_of: date | None = None
    cash: dict[str, float] = field(default_factory=dict)
    positions: dict[str, PositionState] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    orders: dict[str, BrokerOrder] = field(default_factory=dict)
    fills: list[BrokerFill] = field(default_factory=list)
    equity: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def next_order_id(self) -> str:
        return f"{self.account_id}-{len(self.orders) + 1:06d}"

    def next_fill_id(self) -> str:
        return f"{self.account_id}-F{len(self.fills) + 1:06d}"

    def add_event(
        self,
        kind: str,
        session_date: date | None,
        detail: str,
        *,
        instrument_id: str | None = None,
        amount: float | None = None,
        currency: str | None = None,
    ) -> dict[str, Any]:
        ev = {
            "event_id": len(self.events) + 1,
            "ts": now(),
            "session_date": session_date,
            "type": kind,
            "instrument_id": instrument_id,
            "detail": detail,
            "amount": amount,
            "currency": currency,
        }
        self.events.append(ev)
        return ev

    def last_equity_rows(self, n: int = 2) -> list[dict[str, Any]]:
        return self.equity[-n:]


CASH_EVENT_TYPES = {"deposit", "fill", "cost", "dividend", "fx_out", "fx_in", "borrow", "adjustment"}


def recompute_cash(state: AccountState) -> dict[str, float]:
    cash: dict[str, float] = defaultdict(float)
    for ev in state.events:
        if ev["type"] in CASH_EVENT_TYPES and ev.get("amount") is not None and ev.get("currency"):
            cash[str(ev["currency"])] += float(ev["amount"])
    return dict(cash)


def recompute_positions(state: AccountState) -> dict[str, float]:
    qty: dict[str, float] = defaultdict(float)
    events = sorted(
        [e for e in state.events if e["type"] == "split"], key=lambda e: (e["session_date"], e["event_id"])
    )
    fills = sorted(state.fills, key=lambda f: (f.session_date, f.fill_id))
    i = 0
    for f in fills:
        while i < len(events) and events[i]["session_date"] <= f.session_date:
            ev = events[i]
            if ev["session_date"] < f.session_date or ev["detail"].endswith("pre-fill"):
                qty[str(ev["instrument_id"])] *= float(ev["amount"])
                i += 1
            else:
                break
        qty[f.instrument_id] += f.signed_qty
    for ev in events[i:]:
        qty[str(ev["instrument_id"])] *= float(ev["amount"])
    return {k: v for k, v in qty.items() if abs(v) > 1e-9}


def ledger_invariants(state: AccountState, tol: float = 1e-6) -> list[str]:
    issues: list[str] = []
    cash = recompute_cash(state)
    for ccy in sorted(set(cash) | set(state.cash)):
        a, b = state.cash.get(ccy, 0.0), cash.get(ccy, 0.0)
        if abs(a - b) > tol:
            issues.append(f"cash {ccy}: state {a:,.4f} vs events {b:,.4f}")
    pos = recompute_positions(state)
    for iid in sorted(set(pos) | {k for k, p in state.positions.items() if abs(p.qty) > 1e-9}):
        a = state.positions[iid].qty if iid in state.positions else 0.0
        b = pos.get(iid, 0.0)
        if abs(a - b) > tol:
            issues.append(f"position {iid}: state {a:g} vs fills {b:g}")
    for o in state.orders.values():
        filled = sum(f.qty for f in state.fills if f.order_id == o.order_id)
        if abs(filled - o.filled_qty) > tol:
            issues.append(f"order {o.order_id}: filled_qty {o.filled_qty:g} vs fills {filled:g}")
    return issues


class Ledger:
    def __init__(self, db: DuckDBStore) -> None:
        self.db = db
        self.db.migrate()

    # ------------------------------------------------------------------ accounts
    def exists(self, account_id: str) -> bool:
        df = self.db.query("SELECT 1 FROM paper_accounts WHERE account_id = ?", [account_id])
        return not df.empty

    def list_accounts(self) -> pd.DataFrame:
        return self.db.query(
            "SELECT account_id, broker, base_currency, initial_cash, created_at, as_of FROM paper_accounts "
            "ORDER BY created_at"
        )

    def create_account(
        self,
        account_id: str,
        *,
        broker: str,
        base_currency: str,
        initial_cash: float,
        as_of: date | None = None,
        config: dict[str, Any] | None = None,
    ) -> AccountState:
        if self.exists(account_id):
            raise StoreError(f"paper account {account_id!r} already exists")
        if initial_cash <= 0:
            raise StoreError("initial_cash must be positive")
        state = AccountState(
            account_id,
            broker,
            base_currency,
            float(initial_cash),
            now(),
            as_of=as_of,
            cash={base_currency: float(initial_cash)},
            config=dict(config or {}),
        )
        state.add_event(
            "deposit", as_of, "initial deposit", amount=float(initial_cash), currency=base_currency
        )
        self.save(state)
        return state

    def delete_account(self, account_id: str) -> None:
        with self.db.write() as con:
            for table in PAPER_TABLES:
                con.execute(f"DELETE FROM {table} WHERE account_id = ?", [account_id])

    # ------------------------------------------------------------------ persistence
    def save(self, state: AccountState) -> None:
        with self.db.write() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                for table in PAPER_TABLES:
                    con.execute(f"DELETE FROM {table} WHERE account_id = ?", [state.account_id])
                con.execute(
                    "INSERT INTO paper_accounts VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        state.account_id,
                        state.broker,
                        state.base_currency,
                        state.initial_cash,
                        state.created_at,
                        state.as_of,
                        json.dumps(state.config, default=str),
                    ],
                )
                for ccy, amt in state.cash.items():
                    con.execute(
                        "INSERT INTO paper_cash VALUES (?, ?, ?)", [state.account_id, ccy, float(amt)]
                    )
                for iid, p in state.positions.items():
                    con.execute(
                        "INSERT INTO paper_positions VALUES (?, ?, ?, ?, ?)",
                        [state.account_id, iid, float(p.qty), float(p.avg_price), p.currency],
                    )
                for iid, px in state.marks.items():
                    con.execute(
                        "INSERT INTO paper_marks VALUES (?, ?, ?, ?)",
                        [state.account_id, iid, float(px), state.as_of],
                    )
                for o in state.orders.values():
                    con.execute(
                        "INSERT INTO paper_orders VALUES (" + ", ".join(["?"] * 19) + ")",
                        [
                            state.account_id,
                            o.order_id,
                            o.instrument_id,
                            o.side.value,
                            float(o.qty),
                            o.order_type.value,
                            o.limit_price,
                            o.stop_price,
                            o.tif.value,
                            o.status.value,
                            float(o.filled_qty),
                            o.avg_fill_price,
                            o.created_session,
                            o.eligible_from,
                            o.tag,
                            o.client_order_id,
                            "; ".join(o.notes),
                            bool(o.triggered),
                            now(),
                        ],
                    )
                for f in state.fills:
                    con.execute(
                        "INSERT INTO paper_fills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            state.account_id,
                            f.fill_id,
                            f.order_id,
                            f.instrument_id,
                            f.side.value,
                            float(f.qty),
                            float(f.price),
                            f.session_date,
                            f.currency,
                            float(f.commission),
                            float(f.spread),
                            float(f.slippage),
                            float(f.taxes),
                            float(f.fx_cost),
                            f.tag,
                        ],
                    )
                for row in state.equity:
                    con.execute(
                        "INSERT INTO paper_equity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            state.account_id,
                            row["session_date"],
                            float(row["equity"]),
                            float(row["cash_base"]),
                            float(row["positions_value"]),
                            float(row.get("gross_exposure", 0.0)),
                            float(row.get("net_exposure", 0.0)),
                            float(row.get("turnover", 0.0)),
                            int(row.get("n_positions", 0)),
                            int(row.get("fills", 0)),
                        ],
                    )
                for ev in state.events:
                    con.execute(
                        "INSERT INTO paper_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            state.account_id,
                            int(ev["event_id"]),
                            ev["ts"],
                            ev.get("session_date"),
                            ev["type"],
                            ev.get("instrument_id"),
                            ev.get("detail"),
                            ev.get("amount"),
                            ev.get("currency"),
                        ],
                    )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def load(self, account_id: str) -> AccountState:
        acct = self.db.query("SELECT * FROM paper_accounts WHERE account_id = ?", [account_id])
        if acct.empty:
            raise StoreError(f"paper account {account_id!r} not found")
        row = acct.iloc[0]
        as_of = row["as_of"]
        state = AccountState(
            account_id,
            str(row["broker"]),
            str(row["base_currency"]),
            float(row["initial_cash"]),
            pd.Timestamp(row["created_at"]).to_pydatetime(),
            as_of=pd.Timestamp(as_of).date() if pd.notna(as_of) else None,
            config=json.loads(row["config"]) if isinstance(row["config"], str) and row["config"] else {},
        )
        q = self.db.query
        for _, r in q(
            "SELECT currency, amount FROM paper_cash WHERE account_id = ?", [account_id]
        ).iterrows():
            state.cash[str(r["currency"])] = float(r["amount"])
        pos = q("SELECT * FROM paper_positions WHERE account_id = ?", [account_id])
        for _, r in pos.iterrows():
            state.positions[str(r["instrument_id"])] = PositionState(
                float(r["qty"]), float(r["avg_price"]), str(r["currency"])
            )
        for _, r in q(
            "SELECT instrument_id, price FROM paper_marks WHERE account_id = ?", [account_id]
        ).iterrows():
            state.marks[str(r["instrument_id"])] = float(r["price"])
        orders = q("SELECT * FROM paper_orders WHERE account_id = ? ORDER BY order_id", [account_id])
        for _, r in orders.iterrows():
            o = BrokerOrder(
                order_id=str(r["order_id"]),
                instrument_id=str(r["instrument_id"]),
                side=str(r["side"]),
                qty=float(r["qty"]),
                order_type=str(r["order_type"]),
                limit_price=_opt_float(r["limit_price"]),
                stop_price=_opt_float(r["stop_price"]),
                tif=str(r["tif"]),
                status=str(r["status"]),
                filled_qty=float(r["filled_qty"]),
                avg_fill_price=_opt_float(r["avg_fill_price"]),
                created_session=_opt_date(r["created_session"]),
                eligible_from=_opt_date(r["eligible_from"]),
                tag=_opt_str(r["tag"]),
                client_order_id=_opt_str(r["client_order_id"]),
                triggered=bool(r["triggered"]),
                notes=[n for n in str(r["notes"] or "").split("; ") if n],
            )
            state.orders[o.order_id] = o
        fills = q(
            "SELECT * FROM paper_fills WHERE account_id = ? ORDER BY session_date, fill_id", [account_id]
        )
        for _, r in fills.iterrows():
            state.fills.append(
                BrokerFill(
                    fill_id=str(r["fill_id"]),
                    order_id=str(r["order_id"]),
                    instrument_id=str(r["instrument_id"]),
                    side=str(r["side"]),
                    qty=float(r["qty"]),
                    price=float(r["price"]),
                    session_date=pd.Timestamp(r["session_date"]).date(),
                    currency=str(r["currency"]),
                    commission=float(r["commission"]),
                    spread=float(r["spread"]),
                    slippage=float(r["slippage"]),
                    taxes=float(r["taxes"]),
                    fx_cost=float(r["fx_cost"]),
                    tag=_opt_str(r["tag"]),
                )
            )
        eq = q("SELECT * FROM paper_equity WHERE account_id = ? ORDER BY session_date", [account_id])
        for _, r in eq.iterrows():
            d = r.to_dict()
            d.pop("account_id", None)
            d["session_date"] = pd.Timestamp(d["session_date"]).date()
            state.equity.append(d)
        ev = q("SELECT * FROM paper_events WHERE account_id = ? ORDER BY event_id", [account_id])
        for _, r in ev.iterrows():
            d = r.to_dict()
            d.pop("account_id", None)
            d["event_id"] = int(d["event_id"])
            d["session_date"] = _opt_date(d["session_date"])
            d["amount"] = _opt_float(d["amount"])
            d["currency"] = _opt_str(d["currency"])
            d["instrument_id"] = _opt_str(d["instrument_id"])
            state.events.append(d)
        return state

    # ------------------------------------------------------------------ frames
    def equity_frame(self, account_id: str) -> pd.DataFrame:
        return self.db.query(
            "SELECT * FROM paper_equity WHERE account_id = ? ORDER BY session_date", [account_id]
        ).drop(columns=["account_id"])


def _opt_float(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def _opt_str(v: Any) -> str | None:
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def _opt_date(v: Any) -> date | None:
    return None if v is None or pd.isna(v) else pd.Timestamp(v).date()
