"""Reconciliation: the ledger against itself (invariants) and against what the broker reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant.paper.broker import Broker
from quant.paper.ledger import AccountState, ledger_invariants


@dataclass
class ReconciliationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        head = "reconciled" if self.ok else f"{len(self.issues)} issue(s)"
        return "\n".join([head] + [f"  - {i}" for i in self.issues])


def reconcile(state: AccountState, broker: Broker | None = None, tol: float = 1e-6) -> ReconciliationReport:
    issues = list(ledger_invariants(state, tol))
    checks: dict[str, Any] = {"ledger_invariants": len(issues) == 0}
    if broker is not None:
        bpos = {p.instrument_id: p.qty for p in broker.positions()}
        lpos = {k: p.qty for k, p in state.positions.items() if abs(p.qty) > 1e-12}
        for iid in sorted(set(bpos) | set(lpos)):
            if abs(bpos.get(iid, 0.0) - lpos.get(iid, 0.0)) > tol:
                issues.append(
                    f"broker position {iid}: broker {bpos.get(iid, 0.0):g} vs ledger {lpos.get(iid, 0.0):g}"
                )
        acct = broker.account()
        for ccy in sorted(set(acct.cash) | set(state.cash)):
            if abs(acct.cash.get(ccy, 0.0) - state.cash.get(ccy, 0.0)) > tol:
                issues.append(
                    f"broker cash {ccy}: broker {acct.cash.get(ccy, 0.0):,.4f} "
                    f"vs ledger {state.cash.get(ccy, 0.0):,.4f}"
                )
        open_b = {o.order_id for o in broker.open_orders()}
        open_l = {o.order_id for o in state.orders.values() if o.is_open}
        if open_b != open_l:
            issues.append(f"open orders differ: broker {sorted(open_b)} vs ledger {sorted(open_l)}")
        checks["broker_positions"] = bpos
        checks["broker_cash"] = dict(acct.cash)
    return ReconciliationReport(ok=not issues, issues=issues, checks=checks)
