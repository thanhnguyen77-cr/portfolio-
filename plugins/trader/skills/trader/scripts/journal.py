#!/usr/bin/env python3
"""Trade journal: log every order and outcome, and measure your real edge.

  journal.py open  --symbol NVDA --qty 9 --entry 152.30 --stop 146.10 --target 164.70 \
                   --setup breakout --order-id abc123 --plan plans/NVDA-2026-11-06.md [--mode paper]
  journal.py close --order-id abc123 --exit 164.70 --reason target [--entry-fill 152.41] [--fees 0.12] [--date ...]
  journal.py cancel --order-id abc123 [--reason unfilled]   # entry never filled: not a trade
  journal.py note  --order-id abc123 --text "moved stop to breakeven after earnings"
  journal.py list  [--open]
  journal.py stats [--setup breakout] [--mode paper]

stats reports win rate, payoff and expectancy in R from *closed live/paper
trades*, which the position sizer can use instead of backtest numbers once
there are enough of them (`kelly.py --win-rate ... --payoff ... --trades ...`).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import traderlib as tl

FIELDS = ["order_id", "symbol", "side", "setup", "mode", "qty", "entry", "stop", "target",
          "opened_at", "exit", "exit_reason", "closed_at", "fees", "pnl", "r_multiple", "plan", "notes"]


def _path(ws: Optional[str]) -> Path:
    return tl.workspace_dir(ws) / "journal.csv"


def read(ws: Optional[str]) -> list[dict]:
    p = _path(ws)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(rows: list[dict], ws: Optional[str]) -> None:
    p = _path(ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def stats(rows: list[dict]) -> dict:
    closed = [r for r in rows if r.get("closed_at") and r.get("r_multiple") not in ("", None)]
    rs = [float(r["r_multiple"]) for r in closed]
    pnl = [float(r["pnl"]) for r in closed if r.get("pnl") not in ("", None)]
    wins = [x for x in rs if x > 0]
    losses = [x for x in rs if x <= 0]
    avg_w, avg_l = tl.mean(wins), tl.mean(losses)
    return tl.clean_floats({
        "closed_trades": len(closed),
        "open_trades": sum(1 for r in rows if not r.get("closed_at")),
        "cancelled_unfilled": sum(1 for r in rows if r.get("closed_at") and r.get("r_multiple") in ("", None)),
        "win_rate": len(wins) / len(rs) if rs else None,
        "avg_win_r": avg_w,
        "avg_loss_r": avg_l,
        "payoff": (avg_w / abs(avg_l)) if (avg_w and avg_l) else None,
        "expectancy_r": tl.mean(rs),
        "total_pnl": sum(pnl),
        "best_r": max(rs) if rs else None,
        "worst_r": min(rs) if rs else None,
    }, 4)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open")
    for k in ("--symbol", "--order-id"):
        o.add_argument(k, required=True)
    for k in ("--qty", "--entry", "--stop"):
        o.add_argument(k, type=float, required=True)
    o.add_argument("--target", type=float)
    o.add_argument("--side", default="buy")
    o.add_argument("--setup", default="")
    o.add_argument("--mode", default="paper")
    o.add_argument("--plan", default="")
    o.add_argument("--date")
    c = sub.add_parser("close")
    c.add_argument("--order-id", required=True)
    c.add_argument("--exit", type=float, required=True)
    c.add_argument("--reason", default="")
    c.add_argument("--fees", type=float, default=0.0)
    c.add_argument("--entry-fill", type=float, help="actual entry fill price, if different from the logged entry")
    c.add_argument("--date")
    cx = sub.add_parser("cancel")
    cx.add_argument("--order-id", required=True)
    cx.add_argument("--reason", default="unfilled")
    cx.add_argument("--date")
    n = sub.add_parser("note")
    n.add_argument("--order-id", required=True)
    n.add_argument("--text", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--open", action="store_true")
    st = sub.add_parser("stats")
    st.add_argument("--setup")
    st.add_argument("--mode")
    args = ap.parse_args(argv)

    rows = read(args.workspace)
    find = lambda oid: next((r for r in rows if r["order_id"] == oid), None)  # noqa: E731

    if args.cmd == "open":
        if find(args.order_id):
            tl.die(f"order {args.order_id} already journaled")
        rows.append({"order_id": args.order_id, "symbol": args.symbol.upper(), "side": args.side,
                     "setup": args.setup, "mode": args.mode, "qty": args.qty, "entry": args.entry,
                     "stop": args.stop, "target": args.target or "", "opened_at": args.date or tl.now_iso(),
                     "plan": args.plan, "notes": ""})
        write(rows, args.workspace)
        print(f"journaled {args.side} {args.qty:g} {args.symbol.upper()} @ {args.entry} (stop {args.stop})")
    elif args.cmd == "close":
        r = find(args.order_id)
        if not r:
            tl.die(f"order {args.order_id} not in journal")
        if args.entry_fill is not None:
            r["entry"] = args.entry_fill
        qty, entry, stop = float(r["qty"]), float(r["entry"]), float(r["stop"])
        sign = 1 if r.get("side", "buy") == "buy" else -1
        pnl = sign * qty * (args.exit - entry) - args.fees
        risk = qty * abs(entry - stop)
        r.update(exit=args.exit, exit_reason=args.reason, closed_at=args.date or tl.now_iso(),
                 fees=args.fees, pnl=round(pnl, 2), r_multiple=round(pnl / risk, 3) if risk else "")
        write(rows, args.workspace)
        print(f"closed {r['symbol']}: P&L ${pnl:,.2f} ({r['r_multiple']} R) — {args.reason}")
    elif args.cmd == "cancel":
        r = find(args.order_id)
        if not r:
            tl.die(f"order {args.order_id} not in journal")
        r.update(exit_reason=args.reason, closed_at=args.date or tl.now_iso(), pnl="", r_multiple="")
        write(rows, args.workspace)
        print(f"{r['symbol']} order {args.order_id} marked {args.reason} (not counted as a trade)")
    elif args.cmd == "note":
        r = find(args.order_id)
        if not r:
            tl.die(f"order {args.order_id} not in journal")
        stamp = tl.now_iso()[:16]
        r["notes"] = (r.get("notes") + " | " if r.get("notes") else "") + f"{stamp} {args.text}"
        write(rows, args.workspace)
        print("noted")
    elif args.cmd == "list":
        sel = [r for r in rows if not r.get("closed_at")] if args.open else rows
        print(tl.dump_json(sel))
    elif args.cmd == "stats":
        sel = [r for r in rows if (not args.setup or r.get("setup") == args.setup)
               and (not args.mode or r.get("mode") == args.mode)]
        print(tl.dump_json(stats(sel)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
