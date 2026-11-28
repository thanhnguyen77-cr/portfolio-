#!/usr/bin/env python3
"""Size a position with a fractional, capped Kelly Criterion.

Kelly for a trade with a hard stop: you either lose 1R (the stop) or win
b x R (the target), with win probability p. The fraction of capital to put
AT RISK is

    f* = p - (1 - p) / b

Full Kelly is far too aggressive for estimated edges, so this script:

  1. shrinks p to the 1-sigma Wilson lower bound for the sample size
     (a 55% win rate over 40 trades is not the same as over 4,000),
  2. multiplies by profile.kelly_fraction (default 0.25 = quarter Kelly),
  3. scales by the market-regime multiplier (risk_on 1.0, caution 0.5, risk_off 0),
  4. caps the dollars at risk at profile.max_risk_per_trade_pct of capital,
  5. caps the position value at profile.max_position_value,
  6. caps the cost at the cash available above profile.cash_reserve,
  7. rounds down to whole shares.

The smallest cap wins and is reported as `binding_constraint`. A negative
Kelly (no edge) returns NO_TRADE.

Examples:
  kelly.py --backtest trader-workspace/backtests/NVDA.json --entry 152.3 --stop 146.1 --target 164.7
  kelly.py --win-rate 0.52 --payoff 2.1 --trades 64 --entry 50 --stop 47.5 --account account.json --regime caution
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import traderlib as tl

REGIME_MULT = {"risk_on": 1.0, "caution": 0.5, "risk_off": 0.0}


def kelly_fraction(p: float, b: float) -> float:
    if b <= 0:
        return -1.0
    return p - (1 - p) / b


def size(*, p_hat: float, payoff: float, n_trades: int, entry: float, stop: float, prof: dict,
         account: Optional[dict] = None, regime: str = "risk_on", conservative: bool = True,
         target: Optional[float] = None) -> dict:
    notes: list[str] = []
    if entry <= 0 or stop <= 0:
        raise ValueError("entry and stop must be positive")
    if stop >= entry:
        raise ValueError("long trades only: stop must be below entry")

    p_used = tl.wilson_lower_bound(p_hat, n_trades, z=1.0) if (conservative and n_trades > 0) else p_hat
    if conservative and n_trades > 0:
        notes.append(f"win rate shrunk from {p_hat:.3f} to {p_used:.3f} (1-sigma lower bound, n={n_trades})")
    f_full = kelly_fraction(p_used, payoff)
    f_raw = kelly_fraction(p_hat, payoff)
    mult = REGIME_MULT.get(regime, 0.5)
    f_used = max(0.0, f_full) * float(prof["kelly_fraction"]) * mult

    budget = float(prof["budget"])
    reserve = float(prof["cash_reserve"])
    capital = budget
    invested = 0.0
    cash = budget
    if account:
        eq = tl._to_float(account.get("equity") or account.get("portfolio_value"))
        if eq:
            capital = min(budget, eq)
        invested = sum(abs(tl._to_float(pp.get("market_value")) or 0.0) for pp in account.get("positions", []))
        acct_cash = tl._to_float(account.get("cash"))
        # Cash this desk may use: its budget minus what is already invested, never more than the account holds
        cash = budget - invested
        if acct_cash is not None:
            cash = min(cash, acct_cash)

    fee_cfg = prof.get("fees", {})
    slip = (float(fee_cfg.get("slippage_bps", 0)) + float(fee_cfg.get("fee_bps", 0))) / 1e4
    risk_per_share = (entry - stop) + entry * slip * 2  # stop distance + round-trip costs

    result = {
        "inputs": {"win_rate": p_hat, "win_rate_used": p_used, "payoff": payoff, "trades": n_trades,
                   "entry": entry, "stop": stop, "target": target, "regime": regime,
                   "kelly_fraction_setting": prof["kelly_fraction"], "regime_multiplier": mult},
        "kelly_full_raw": f_raw,
        "kelly_full_conservative": f_full,
        "kelly_used": f_used,
        "capital_base": capital,
        "risk_per_share": risk_per_share,
    }

    if f_full <= 0:
        result.update(decision="NO_TRADE", qty=0, binding_constraint="kelly",
                      notes=notes + [f"Kelly is {f_full:.3f} ≤ 0 — the measured edge does not justify any risk"])
        return result
    if mult == 0:
        result.update(decision="NO_TRADE", qty=0, binding_constraint="regime",
                      notes=notes + ["market regime is risk_off — no new longs"])
        return result

    risk_cap_dollars = capital * float(prof["max_risk_per_trade_pct"]) / 100.0
    kelly_risk_dollars = capital * f_used
    deployable = cash - reserve
    caps = {
        "kelly": math.floor(kelly_risk_dollars / risk_per_share),
        "max_risk_per_trade": math.floor(risk_cap_dollars / risk_per_share),
        "max_position_value": math.floor(float(prof["max_position_value"]) / entry),
        "cash_reserve": math.floor(max(0.0, deployable) / entry),
    }
    binding = min(caps, key=lambda k: caps[k])
    qty = max(0, caps[binding])

    exposure_cap = budget * float(prof.get("max_total_exposure_pct", 100)) / 100.0
    room = exposure_cap - invested
    exp_qty = math.floor(max(0.0, room) / entry)
    if exp_qty < qty:
        qty, binding = exp_qty, "max_total_exposure"
        caps["max_total_exposure"] = exp_qty

    result.update({
        "caps_qty": caps,
        "binding_constraint": binding,
        "qty": qty,
        "position_value": qty * entry,
        "dollars_at_risk": qty * risk_per_share,
        "risk_pct_of_capital": (qty * risk_per_share / capital * 100) if capital else None,
        "cash_after": cash - qty * entry,
        "cash_reserve": reserve,
    })
    if target and target > entry:
        result["reward_risk"] = (target - entry) / (entry - stop)
        result["potential_profit"] = qty * (target - entry)
    if qty < 1:
        result["decision"] = "NO_TRADE"
        notes.append(f"position rounds to 0 shares (binding: {binding}); "
                     "the stop is too wide or the limits too tight for this price")
    else:
        result["decision"] = "SIZE"
        if binding != "kelly":
            notes.append(f"Kelly would allow {caps['kelly']} shares; capped at {qty} by {binding}")
    result["notes"] = notes
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backtest", help="backtest JSON (uses kelly_inputs)")
    ap.add_argument("--win-rate", type=float)
    ap.add_argument("--payoff", type=float, help="average win / average loss (in R)")
    ap.add_argument("--trades", type=int, default=0, help="sample size behind the win rate")
    ap.add_argument("--entry", type=float)
    ap.add_argument("--stop", type=float)
    ap.add_argument("--target", type=float)
    ap.add_argument("--plan", help="plan JSON; fills entry/stop/target if not given")
    ap.add_argument("--profile")
    ap.add_argument("--account", help="account JSON: {equity, cash, positions:[{symbol, market_value}]}")
    ap.add_argument("--regime", choices=sorted(REGIME_MULT), default="risk_on")
    ap.add_argument("--no-shrink", action="store_true", help="use the raw win rate (not recommended)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    prof = tl.load_profile(args.profile)
    p_hat, payoff, n = args.win_rate, args.payoff, args.trades
    if args.backtest:
        k = tl.load_json(args.backtest).get("kelly_inputs", {})
        p_hat = p_hat if p_hat is not None else k.get("win_rate")
        payoff = payoff if payoff is not None else k.get("payoff")
        n = n or int(k.get("trades") or 0)
    if p_hat is None or payoff is None:
        tl.die("need --win-rate and --payoff (or --backtest with kelly_inputs)")
    if args.plan:
        plan = tl.load_json(args.plan)
        # size on the worst fill: a stop-limit entry can fill as high as its limit
        worst = plan.get("entry_limit") if plan.get("entry_type") == "stop_limit" else None
        args.entry = args.entry if args.entry is not None else tl._to_float(worst or plan.get("entry"))
        args.stop = args.stop if args.stop is not None else tl._to_float(plan.get("stop"))
        args.target = args.target if args.target is not None else tl._to_float(plan.get("target"))
    if args.entry is None or args.stop is None:
        tl.die("need --entry and --stop (or --plan)")
    account = tl.load_json(args.account) if args.account else None
    try:
        res = size(p_hat=float(p_hat), payoff=float(payoff), n_trades=n, entry=args.entry, stop=args.stop,
                   prof=prof, account=account, regime=args.regime, conservative=not args.no_shrink,
                   target=args.target)
    except ValueError as e:
        tl.die(str(e))
    res = tl.clean_floats(res, 4)
    res["generated_at"] = tl.now_iso()
    text = tl.dump_json(res, args.out)
    if args.out:
        print(f"{res['decision']}: {res.get('qty', 0)} shares "
              f"(${res.get('position_value', 0):,.2f}, risk ${res.get('dollars_at_risk', 0):,.2f}); "
              f"binding: {res.get('binding_constraint')}")
        for nt in res.get("notes", []):
            print(f"  - {nt}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
