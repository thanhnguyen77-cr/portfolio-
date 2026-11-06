#!/usr/bin/env python3
"""PreToolUse hook: block broker orders that the risk manager did not approve.

Claude Code runs this before any tool whose name matches the hook matcher
(the Alpaca MCP tools). It reads the hook payload on stdin:

  {"tool_name": "mcp__alpaca__place_stock_order",
   "tool_input": {...}, "cwd": "/path/to/project", ...}

Read-only tools, cancels and position closes pass straight through (they
never add risk). Anything that opens or modifies an order must match a
single-use, unexpired approval ticket written by risk_check.py --approve:
same symbol and side, qty no larger than approved, limit price no worse than
approved (within profile.guard.max_limit_slippage_pct), and for entries a
protective stop must be attached. A matching ticket is marked consumed.

Exit 0 = allow, exit 2 = block (stderr is shown to Claude as the reason).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import traderlib as tl  # noqa: E402

# Tool-name suffixes that place or change orders (Alpaca MCP v2 names + common aliases).
ORDER_TOOLS = re.compile(r"(place_|submit_|create_).*order|replace_order|^order$|exercise", re.I)
SAFE_TOOLS = re.compile(r"^(get_|list_|cancel_|close_)", re.I)


def block(msg: str) -> None:
    print(f"[trader order guard] BLOCKED: {msg}", file=sys.stderr)
    sys.exit(2)


def _num(d: dict, *keys: str) -> Optional[float]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return tl._to_float(d[k])
    return None


def _walk(d: Any, prefix: str = ""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, f"{prefix}{k}.".lower())
            yield f"{prefix}{k}".lower(), v


def has_protective_stop(inp: dict) -> bool:
    oc = str(inp.get("order_class", "")).lower()
    if oc in ("bracket", "oto"):
        return True
    for key, val in _walk(inp):
        if ("stop_loss" in key or key.split(".")[-1] in ("sl", "sl_price")) and val not in (None, "", {}, []):
            return True
    return False


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return 0  # not our business if we cannot parse; Claude Code will still prompt the user
    tool = str(payload.get("tool_name", ""))
    action = tool.split("__")[-1]
    if SAFE_TOOLS.match(action) or not ORDER_TOOLS.search(action):
        return 0

    inp = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or os.getcwd())
    ws = Path(os.environ.get(tl.WORKSPACE_ENV) or (cwd / tl.DEFAULT_WORKSPACE))
    prof_path = ws / "profile.json"
    if not prof_path.exists():
        block(f"no trader profile at {prof_path}. Set up the trader desk (budget, cash reserve, max per stock) first.")
    prof = tl.merge_defaults(tl.load_json(prof_path))
    guard = prof.get("guard") or {}
    if prof.get("mode") == "live" and not prof.get("live_trading_enabled"):
        block("profile is in live mode but live_trading_enabled is false")

    if "stock" not in action and "replace" not in action and action not in ("place_order", "submit_order"):
        block(f"{action}: the trader desk only trades stocks/ETFs; options/crypto orders are not approved")

    sym = str(inp.get("symbol") or inp.get("ticker") or "").upper()
    side = str(inp.get("side") or "").lower()
    qty = _num(inp, "qty", "quantity", "shares")
    if "replace" not in action:
        if _num(inp, "notional") and not qty:
            block("notional (dollar-amount) orders are not allowed; send a whole-share qty")
        if not sym or not side or not qty:
            block(f"order is missing symbol/side/qty (got symbol={sym!r}, side={side!r}, qty={qty})")
    limit_px = _num(inp, "limit_price", "price")
    order_id = str(inp.get("order_id") or inp.get("id") or "")

    appr_dir = ws / "risk" / "approvals"
    now = datetime.now(timezone.utc)
    max_slip = float(guard.get("max_limit_slippage_pct", 0.5)) / 100
    reasons: list[str] = []
    for f in sorted(appr_dir.glob("*.json"), reverse=True) if appr_dir.exists() else []:
        try:
            t = tl.load_json(f)
        except Exception:  # noqa: BLE001
            continue
        if t.get("status") != "APPROVED" or t.get("consumed_at"):
            continue
        exp = tl.parse_date(t.get("expires_at", ""))
        if not exp or exp < now:
            reasons.append(f"{t.get('id')}: expired")
            continue
        if "replace" in action:
            if t.get("kind") != "replace" or str(t.get("replace_order_id")) != order_id:
                continue
        else:
            if t.get("symbol") != sym or t.get("side") != side:
                continue
        if qty and t.get("qty") and qty > float(t["qty"]) + 1e-9:
            reasons.append(f"{t['id']}: qty {qty} > approved {t['qty']}")
            continue
        ap_lim = t.get("limit_price")
        if limit_px and ap_lim:
            worse = limit_px > ap_lim * (1 + max_slip) if side == "buy" else limit_px < ap_lim * (1 - max_slip)
            if worse:
                reasons.append(f"{t['id']}: limit {limit_px} is worse than approved {ap_lim}")
                continue
        if t.get("kind") == "entry" and guard.get("require_bracket", True) and not has_protective_stop(inp):
            reasons.append(f"{t['id']}: entry order has no protective stop (use a bracket/OTO order with "
                           f"stop loss {t.get('stop_loss')})")
            continue
        t["consumed_at"] = now.replace(microsecond=0).isoformat()
        t["status"] = "USED"
        t["used_by_tool"] = tool
        tl.dump_json(t, f)
        print(f"[trader order guard] allowed by approval {t['id']}", file=sys.stderr)
        return 0

    detail = ("; ".join(reasons[:5])) if reasons else "no approval ticket found"
    block(f"{action} {side} {qty} {sym}: {detail}. Run the risk manager "
          f"(risk_check.py --approve) and use the approved parameters.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
