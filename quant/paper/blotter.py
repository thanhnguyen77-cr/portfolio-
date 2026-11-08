"""Blotter: the paper account's state as a self-contained HTML report (account, positions, orders, fills,
rejections, equity curve) with the mandatory caveats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.paper.broker import Broker
from quant.paper.ledger import AccountState
from quant.paper.reconcile import ReconciliationReport
from quant.reports.builder import Section, plotly_js_source, render_report


def _equity_chart(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(rows) < 2:
        return None
    x = [str(r["session_date"]) for r in rows]
    y = [float(r["equity"]) for r in rows]
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "name": "equity",
                "x": x,
                "y": y,
                "hovertemplate": "%{y:,.2f}<extra>equity</extra>",
                "meta": {"role": "series", "slot": 0},
            }
        ],
        "layout": {"yaxis": {"tickformat": ",.0f"}},
    }


def build_blotter(
    state: AccountState,
    broker: Broker | None = None,
    *,
    reconciliation: ReconciliationReport | None = None,
    pretrade_rejections: pd.DataFrame | None = None,
    title: str | None = None,
    path: Path | str | None = None,
    inline_plotly: bool = False,
    extra_caveats: list[str] | None = None,
) -> str:
    acct = broker.account().as_dict() if broker is not None else {
        "account_id": state.account_id, "broker": state.broker, "as_of": state.as_of,
        "base_currency": state.base_currency, "cash": state.cash,
    }  # fmt: skip
    kv = {k: (f"{v:,.2f}" if isinstance(v, float) else v) for k, v in acct.items() if k != "cash"}
    kv["cash"] = ", ".join(f"{c} {a:,.2f}" for c, a in acct["cash"].items())
    sections = [Section("Account", kv=kv, chart=_equity_chart(state.equity))]
    positions = (
        pd.DataFrame([p.as_dict() for p in broker.positions()])
        if broker is not None
        else pd.DataFrame(
            [{"instrument_id": k, **p.__dict__} for k, p in state.positions.items() if abs(p.qty) > 1e-12]
        )
    )
    sections.append(
        Section(
            "Positions",
            table=positions if not positions.empty else None,
            text=None if not positions.empty else "no open positions",
        )
    )
    orders = pd.DataFrame([o.as_dict() for o in state.orders.values()])
    open_orders = orders[orders["status"].isin(["open", "partially_filled"])] if not orders.empty else orders
    sections.append(
        Section(
            "Open orders",
            table=open_orders if not open_orders.empty else None,
            text=None if not open_orders.empty else "none",
        )
    )
    fills = pd.DataFrame([f.as_dict() for f in state.fills])
    sections.append(
        Section(
            "Fills (latest 100)",
            table=fills.sort_values(["session_date", "fill_id"]).tail(100) if not fills.empty else None,
            text=None if not fills.empty else "no fills yet",
            kv={"total costs": f"{fills['total_cost'].sum():,.2f} (local ccy, summed)"}
            if not fills.empty
            else None,
        )
    )
    rejected = orders[orders["status"] == "rejected"] if not orders.empty else orders
    if pretrade_rejections is not None and not pretrade_rejections.empty:
        sections.append(Section("Blocked by pre-trade rules", table=pretrade_rejections))
    if not rejected.empty:
        sections.append(Section("Rejected by broker", table=rejected))
    events = pd.DataFrame(state.events)
    if not events.empty:
        sections.append(Section("Events (latest 50)", table=events.tail(50)))
    if reconciliation is not None:
        sections.append(
            Section(
                "Reconciliation",
                text=reconciliation.summary(),
                kv={"ok": reconciliation.ok, "issues": len(reconciliation.issues)},
            )
        )
    caveats = [
        "PAPER TRADING. Fills are simulated against daily bars at the next session open with the shared cost "
        "model; real fills, queue position, halts and outages are not modelled.",
        "Marks use the last cached close; stale data makes equity and buying power stale too.",
        "No live broker exists in this version: live trading is disabled by design.",
    ] + list(extra_caveats or [])
    if state.equity:
        eq = np.array([r["equity"] for r in state.equity], dtype=float)
        caveats.append(
            f"Since inception: {eq[-1] / state.initial_cash - 1:+.2%} on {len(eq)} marked session(s); "
            "far too "
            "short to say anything about skill."
        )
    subtitle = f"{state.account_id} · {state.broker} · {state.base_currency} · as of {state.as_of}"
    return render_report(
        title or f"Paper blotter: {state.account_id}",
        sections,
        caveats,
        subtitle=subtitle,
        path=path,
        inline_js=plotly_js_source() if inline_plotly else None,
    )
