#!/usr/bin/env python3
"""Pre-trade risk check. The risk manager's veto, as code.

Every order the broker agent wants to send is checked here first against the
profile limits, the live account state, the backtest verdict, the news desk's
flags and the market regime. Exit status 0 = APPROVE, 2 = REJECT.

With --approve, an APPROVE writes a single-use approval ticket to
trader-workspace/risk/approvals/<id>.json. The order guard hook
(order_guard.py) refuses any broker order that has no matching, unexpired,
unused ticket — so an order cannot skip this check.

Order JSON (entry):
  {"symbol": "NVDA", "side": "buy", "qty": 9, "order_type": "limit",
   "limit_price": 152.30, "stop_loss": 146.10, "take_profit": 164.70,
   "time_in_force": "day"}
  order_type: limit | stop_limit (needs stop_price trigger) | market (only if allowed)

Order JSON (exit, with --exit):
  {"symbol": "NVDA", "side": "sell", "qty": 9, "order_type": "limit", "limit_price": 160.0}

Account JSON (from the broker's account + positions tools):
  {"equity": 10250, "last_equity": 10180, "cash": 6100,
   "positions": [{"symbol": "AAPL", "qty": 5, "market_value": 950}],
   "open_orders": [{"symbol": "MSFT", "side": "buy", "qty": 3, "limit_price": 410}]}

News JSON (from the news desk, optional):
  {"symbol": "NVDA", "severity": "green|amber|red", "days_to_earnings": 12, "flags": ["..."]}

Example:
  risk_check.py --order order.json --account account.json --backtest bt.json \
                --sizing sizing.json --news news.json --regime risk_on --approve
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import traderlib as tl


class Checks:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, name: str, status: str, detail: str) -> None:
        self.items.append({"name": name, "status": status, "detail": detail})

    def ok(self, name: str, cond: bool, detail: str, fail_status: str = "fail") -> None:
        self.add(name, "pass" if cond else fail_status, detail)

    @property
    def failed(self) -> list[dict]:
        return [c for c in self.items if c["status"] == "fail"]

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.items if c["status"] == "warn"]


def _positions(account: Optional[dict]) -> list[dict]:
    return list((account or {}).get("positions") or [])


def _held_qty(account: Optional[dict], symbol: str) -> float:
    for p in _positions(account):
        if str(p.get("symbol", "")).upper() == symbol:
            return abs(tl._to_float(p.get("qty")) or 0.0)
    return 0.0


def _held_value(account: Optional[dict], symbol: str) -> float:
    for p in _positions(account):
        if str(p.get("symbol", "")).upper() == symbol:
            return abs(tl._to_float(p.get("market_value")) or 0.0)
    return 0.0


def check_entry(order: dict, prof: dict, account: Optional[dict], backtest: Optional[dict],
                sizing: Optional[dict], news: Optional[dict], regime: Optional[str]) -> Checks:
    ck = Checks()
    sym = str(order.get("symbol", "")).upper()
    side = str(order.get("side", "")).lower()
    qty = tl._to_float(order.get("qty")) or 0.0
    otype = str(order.get("order_type") or order.get("type") or "").lower()
    limit_px = tl._to_float(order.get("limit_price"))
    trigger_px = tl._to_float(order.get("stop_price"))
    stop = tl._to_float(order.get("stop_loss"))
    target = tl._to_float(order.get("take_profit"))
    # worst fill (sets cost and $ at risk) vs planned entry (sets reward:risk).
    # For a stop-limit buy the plan's entry is the trigger; the fill can be as high as the limit.
    entry = limit_px or trigger_px or tl._to_float(order.get("ref_price"))
    planned = trigger_px if (otype == "stop_limit" and trigger_px) else entry

    # --- mode -------------------------------------------------------------
    if prof["mode"] == "paper":
        ck.add("mode", "pass", "paper trading")
    else:
        ck.ok("mode", bool(prof.get("live_trading_enabled")) and bool(order.get("live_confirmed_by_user")),
              "LIVE mode requires profile.live_trading_enabled AND order.live_confirmed_by_user")

    # --- order shape ------------------------------------------------------
    ck.ok("symbol", bool(sym) and sym.replace(".", "").replace("-", "").isalnum(), f"symbol {sym!r}")
    ck.ok("blocklist", sym not in {s.upper() for s in prof.get("blocklist") or []}, f"{sym} not blocklisted")
    if side == "buy":
        ck.add("side", "pass", "long entry")
    elif side == "sell" and prof.get("allow_shorts"):
        ck.add("side", "warn", "short entry (allowed by profile)")
    else:
        ck.add("side", "fail", f"side {side!r} not allowed for entries (shorts disabled)")
    ck.ok("qty", qty >= 1 and float(qty).is_integer(), f"qty {qty} (whole shares, >= 1)")

    allowed_types = {"limit", "stop_limit"} | ({"market"} if prof.get("allow_market_orders") else set())
    ck.ok("order_type", otype in allowed_types, f"{otype or 'missing'} (allowed: {', '.join(sorted(allowed_types))})")
    if otype == "stop_limit":
        ck.ok("stop_limit_prices", bool(trigger_px and limit_px and limit_px >= trigger_px),
              f"trigger {trigger_px}, limit {limit_px} (limit must be >= trigger for buys)")
    ck.ok("entry_price", bool(entry and entry > 0), f"entry reference {entry}")
    if not entry:
        return ck

    ck.ok("protective_stop", bool(stop and stop < planned), f"stop_loss {stop} below entry {planned}")
    ck.ok("take_profit", bool(target and target > entry), f"take_profit {target} above worst fill {entry}")
    if stop and target and stop < planned < target:
        rr = (target - planned) / (planned - stop)
        ck.ok("reward_risk", rr >= float(prof["min_reward_risk"]) - 1e-9,
              f"{rr:.2f} R from planned entry {planned} vs min {prof['min_reward_risk']}")
    ck.ok("min_price", entry >= float(prof["min_price"]), f"{entry} vs min {prof['min_price']}")

    # --- money limits -----------------------------------------------------
    budget = float(prof["budget"])
    capital = budget
    if account and tl._to_float(account.get("equity")):
        capital = min(budget, tl._to_float(account["equity"]))
    cost = qty * entry
    existing = _held_value(account, sym)
    ck.ok("max_position_value", cost + existing <= float(prof["max_position_value"]) + 1e-6,
          f"${cost + existing:,.2f} in {sym} vs max ${float(prof['max_position_value']):,.2f}")
    if stop and stop < entry:
        risk = qty * (entry - stop)
        cap = capital * float(prof["max_risk_per_trade_pct"]) / 100
        ck.ok("max_risk_per_trade", risk <= cap * 1.001,
              f"${risk:,.2f} at risk (from worst fill {entry}) vs max ${cap:,.2f} "
              f"({prof['max_risk_per_trade_pct']}% of ${capital:,.0f})")

    if account is None:
        ck.add("account_state", "fail", "no account snapshot — fetch account + positions from the broker first")
    else:
        invested = sum(abs(tl._to_float(p.get("market_value")) or 0.0) for p in _positions(account))
        pending = sum((tl._to_float(o.get("qty")) or 0) * (tl._to_float(o.get("limit_price")) or 0)
                      for o in account.get("open_orders") or [] if str(o.get("side", "")).lower() == "buy")
        desk_cash = budget - invested - pending
        acct_cash = tl._to_float(account.get("cash"))
        if acct_cash is not None:
            desk_cash = min(desk_cash, acct_cash - pending)
        reserve = float(prof["cash_reserve"])
        ck.ok("cash_reserve", desk_cash - cost >= reserve - 1e-6,
              f"cash after ${desk_cash - cost:,.2f} vs reserve ${reserve:,.2f}")
        exp_cap = budget * float(prof["max_total_exposure_pct"]) / 100
        ck.ok("total_exposure", invested + pending + cost <= exp_cap + 1e-6,
              f"exposure after ${invested + pending + cost:,.2f} vs cap ${exp_cap:,.2f}")
        held_syms = {str(p.get("symbol", "")).upper() for p in _positions(account)}
        n_after = len(held_syms | {sym})
        ck.ok("max_open_positions", n_after <= int(prof["max_open_positions"]),
              f"{n_after} positions after vs max {prof['max_open_positions']}")
        dup_order = any(str(o.get("symbol", "")).upper() == sym and str(o.get("side", "")).lower() == side
                        for o in account.get("open_orders") or [])
        ck.ok("no_duplicate", sym not in held_syms and not dup_order,
              f"no existing position or open {side} order in {sym}")
        eq, last = tl._to_float(account.get("equity")), tl._to_float(account.get("last_equity"))
        if eq and last:
            day = (eq / last - 1) * 100
            ck.ok("daily_loss_limit", day > -float(prof["max_daily_loss_pct"]),
                  f"today {day:+.2f}% vs limit -{prof['max_daily_loss_pct']}%")
        else:
            ck.add("daily_loss_limit", "warn", "last_equity missing; daily loss not checked")

    # --- evidence ---------------------------------------------------------
    if prof.get("require_backtest", True):
        if not backtest:
            ck.add("backtest", "fail", "no backtest supplied (profile.require_backtest is on)")
        else:
            v = backtest.get("verdict")
            syms = [s.upper() for s in backtest.get("symbols") or []]
            ck.ok("backtest", v == "PASS", f"verdict {v} ({(backtest.get('all') or {}).get('trades')} trades)")
            ck.ok("backtest_symbol", sym in syms, f"{sym} in backtested symbols {syms}")
    if sizing:
        sq = tl._to_float(sizing.get("qty")) or 0
        ck.ok("sizing", sizing.get("decision") == "SIZE" and qty <= sq,
              f"qty {qty} vs position sizer {sq} ({sizing.get('decision')})")
    else:
        ck.add("sizing", "warn", "no sizing file supplied")
    if regime:
        if regime == "risk_off":
            ck.add("regime", "fail", "market regime is risk_off — no new longs")
        elif regime == "caution":
            ck.add("regime", "warn", "market regime is caution — size should be halved")
        else:
            ck.add("regime", "pass", regime)
    else:
        ck.add("regime", "warn", "no market-regime read supplied")
    if news:
        sev = str(news.get("severity", "")).lower()
        if sev == "red":
            ck.add("news", "fail", "news desk flagged RED: " + "; ".join(news.get("flags") or []))
        elif sev == "amber":
            ck.add("news", "warn", "news desk flagged AMBER: " + "; ".join(news.get("flags") or []))
        else:
            ck.add("news", "pass", "no material news flags")
        dte = news.get("days_to_earnings")
        if isinstance(dte, (int, float)) and dte >= 0:
            if dte <= 2:
                ck.add("earnings", "fail", f"earnings in {dte} days — binary event risk")
            elif backtest and dte <= (backtest.get("params") or {}).get("max_hold", 20):
                ck.add("earnings", "warn", f"earnings in {dte} days, inside the planned holding period")
    else:
        ck.add("news", "warn", "no news check supplied")
    return ck


def check_exit(order: dict, prof: dict, account: Optional[dict]) -> Checks:
    ck = Checks()
    sym = str(order.get("symbol", "")).upper()
    side = str(order.get("side", "")).lower()
    qty = tl._to_float(order.get("qty")) or 0.0
    otype = str(order.get("order_type") or order.get("type") or "").lower()
    ck.ok("side", side == "sell", "exits of long positions are sells")
    if account is None:
        ck.add("account_state", "fail", "no account snapshot — fetch positions first")
        return ck
    held = _held_qty(account, sym)
    ck.ok("position_exists", held > 0, f"holding {held} {sym}")
    ck.ok("qty", 0 < qty <= held, f"sell {qty} of {held} held (no flipping short)")
    ck.ok("order_type", otype in {"limit", "market", "stop", "stop_limit"}, f"{otype}")
    return ck


def write_approval(ws: Path, order: dict, kind: str, prof: dict, checks: Checks) -> dict:
    created = datetime.now(timezone.utc).replace(microsecond=0)
    ttl = int(prof.get("approval_ttl_minutes", 30))
    sym = str(order.get("symbol", "")).upper()
    digest = hashlib.sha1(tl.dump_json(order).encode()).hexdigest()[:8]
    aid = f"{sym}-{created.strftime('%Y%m%dT%H%M%S')}-{digest}"
    ticket = {
        "id": aid,
        "status": "APPROVED",
        "kind": kind,
        "symbol": sym,
        "side": str(order.get("side", "")).lower(),
        "qty": tl._to_float(order.get("qty")),
        "order_type": order.get("order_type") or order.get("type"),
        "limit_price": tl._to_float(order.get("limit_price")),
        "stop_price": tl._to_float(order.get("stop_price")),
        "stop_loss": tl._to_float(order.get("stop_loss")),
        "take_profit": tl._to_float(order.get("take_profit")),
        "replace_order_id": order.get("replace_order_id"),
        "mode": prof["mode"],
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(minutes=ttl)).isoformat(),
        "consumed_at": None,
        "warnings": [c["detail"] for c in checks.warnings],
    }
    tl.dump_json(ticket, ws / "risk" / "approvals" / f"{aid}.json")
    return ticket


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", required=True, help="order JSON file")
    ap.add_argument("--account", help="account JSON file")
    ap.add_argument("--backtest")
    ap.add_argument("--sizing")
    ap.add_argument("--news")
    ap.add_argument("--regime", choices=["risk_on", "caution", "risk_off"])
    ap.add_argument("--exit", action="store_true", help="this order closes/reduces an existing long")
    ap.add_argument("--profile")
    ap.add_argument("--workspace")
    ap.add_argument("--approve", action="store_true", help="write a single-use approval ticket on APPROVE")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    prof = tl.load_profile(args.profile, args.workspace)
    order = tl.load_json(args.order)
    account = tl.load_json(args.account) if args.account else None
    ld = lambda p: tl.load_json(p) if p else None  # noqa: E731
    if args.exit:
        ck = check_exit(order, prof, account)
        kind = "exit"
    else:
        ck = check_entry(order, prof, account, ld(args.backtest), ld(args.sizing), ld(args.news), args.regime)
        kind = "replace" if order.get("replace_order_id") else "entry"

    decision = "APPROVE" if not ck.failed else "REJECT"
    res = {
        "generated_at": tl.now_iso(),
        "decision": decision,
        "kind": kind,
        "order": order,
        "failed": [c["name"] for c in ck.failed],
        "warnings": [c["name"] for c in ck.warnings],
        "checks": ck.items,
    }
    if decision == "APPROVE" and args.approve:
        res["approval"] = write_approval(tl.workspace_dir(args.workspace), order, kind, prof, ck)
    text = tl.dump_json(tl.clean_floats(res, 4), args.out)
    if args.out:
        print(f"{decision} {order.get('side')} {order.get('qty')} {order.get('symbol')}"
              + (f" — failed: {', '.join(res['failed'])}" if res["failed"] else "")
              + (f" — warnings: {', '.join(res['warnings'])}" if res["warnings"] else "")
              + (f" — ticket {res['approval']['id']}" if res.get("approval") else ""))
    else:
        print(text)
    return 0 if decision == "APPROVE" else 2


if __name__ == "__main__":
    sys.exit(main())
