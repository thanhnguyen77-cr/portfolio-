#!/usr/bin/env python3
"""Create, show, update and validate the trader profile (your money rules).

  trader_profile.py init --budget 10000 --cash-reserve 2000 --max-position 1500 [--risk-per-trade 1] [--force]
  trader_profile.py show
  trader_profile.py set max_open_positions=4 screen.rsi_max=68 kelly_fraction=0.2
  trader_profile.py validate

The profile lives at trader-workspace/profile.json (override with
--workspace or the TRADER_WORKSPACE environment variable). Live trading cannot be switched on with this script: `mode` and
`live_trading_enabled` must be edited in profile.json by the human, by hand,
and the broker connector must separately be pointed at a live account.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional

import traderlib as tl


LIVE_KEYS = {"mode", "live_trading_enabled"}


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _summary(p: dict) -> str:
    deployable = p["budget"] - p["cash_reserve"]
    risk_d = p["budget"] * p["max_risk_per_trade_pct"] / 100
    return "\n".join([
        f"Budget            ${p['budget']:,.2f} {p['currency']}",
        f"Cash kept aside   ${p['cash_reserve']:,.2f}  (deployable ${deployable:,.2f})",
        f"Max per stock     ${p['max_position_value']:,.2f}",
        f"Max risk/trade    {p['max_risk_per_trade_pct']}% (${risk_d:,.2f} if the stop is hit)",
        f"Kelly fraction    {p['kelly_fraction']} of full Kelly",
        f"Open positions    max {p['max_open_positions']} · exposure max {p['max_total_exposure_pct']}% "
        f"· daily loss stop {p['max_daily_loss_pct']}%",
        f"Min reward:risk   {p['min_reward_risk']}",
        f"Mode              {p['mode'].upper()}" + ("" if p['mode'] == 'paper' else
                                                    f" (live enabled: {p['live_trading_enabled']})"),
    ])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--budget", type=float, required=True)
    i.add_argument("--cash-reserve", type=float, required=True)
    i.add_argument("--max-position", type=float, required=True)
    i.add_argument("--risk-per-trade", type=float, help="max %% of capital at risk per trade (default 1)")
    i.add_argument("--kelly-fraction", type=float, help="fraction of full Kelly (default 0.25)")
    i.add_argument("--max-positions", type=int)
    i.add_argument("--currency")
    i.add_argument("--force", action="store_true", help="overwrite an existing profile")
    sub.add_parser("show")
    sub.add_parser("validate")
    s = sub.add_parser("set")
    s.add_argument("pairs", nargs="+", help="key=value (dotted keys for nested, JSON values)")
    args = ap.parse_args(argv)

    ws = tl.workspace_dir(args.workspace)
    path = ws / "profile.json"

    if args.cmd == "init":
        if path.exists() and not args.force:
            tl.die(f"{path} already exists (use --force to overwrite, or `set` to change values)")
        prof = tl.merge_defaults({})
        prof.update(budget=args.budget, cash_reserve=args.cash_reserve, max_position_value=args.max_position)
        if args.risk_per_trade is not None:
            prof["max_risk_per_trade_pct"] = args.risk_per_trade
        if args.kelly_fraction is not None:
            prof["kelly_fraction"] = args.kelly_fraction
        if args.max_positions is not None:
            prof["max_open_positions"] = args.max_positions
        if args.currency:
            prof["currency"] = args.currency
        prof["created_at"] = tl.now_iso()
        errs = tl.validate_profile(prof)
        if errs:
            tl.die("; ".join(errs))
        tl.dump_json(prof, path)
        print(f"profile written to {path}\n\n{_summary(prof)}")
        return 0

    if not path.exists():
        tl.die(f"no profile at {path}; run `trader_profile.py init` first")
    prof = tl.merge_defaults(tl.load_json(path))

    if args.cmd == "show":
        print(_summary(prof) if not tl.validate_profile(prof) else tl.dump_json(prof))
        return 0
    if args.cmd == "validate":
        errs = tl.validate_profile(prof)
        if errs:
            print("INVALID:\n- " + "\n- ".join(errs))
            return 2
        print("OK\n" + _summary(prof))
        return 0
    if args.cmd == "set":
        for pair in args.pairs:
            if "=" not in pair:
                tl.die(f"expected key=value, got {pair!r}")
            key, raw = pair.split("=", 1)
            if key in LIVE_KEYS:
                tl.die(f"{key} cannot be changed from here. Live trading is switched on only by the human, "
                       f"by editing {path} by hand.")
            val = _parse_value(raw)
            node = prof
            parts = key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = val
        errs = tl.validate_profile(prof)
        if errs:
            tl.die("not saved — " + "; ".join(errs))
        prof["updated_at"] = tl.now_iso()
        tl.dump_json(prof, path)
        print("saved\n" + _summary(prof))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
