#!/usr/bin/env python3
"""Backtest a trade plan's setup on historical bars, net of fees and slippage.

A trade plan is a one-off (buy XYZ at 152.30, stop 146.10, target 164.70).
To test it we turn it into the *rule* that produced it and replay that rule
over years of history:

  setup      what triggers an entry (breakout / pullback /
             trend_continuation / mean_reversion), evaluated on the bar's close
  entry      next bar's open, plus slippage
  stop       entry - stop_atr x ATR14 (or stop_pct %), taken from the plan
  target     entry + rr x (entry - stop), taken from the plan
  time stop  exit at the close after max_hold bars

Conservative fill rules: if a bar touches both stop and target, the stop is
assumed to fill first; a gap through the stop fills at the open; one position
at a time per symbol; no look-ahead (signals use only data up to that close).

Results include in-sample vs out-of-sample (last --oos-frac of the history),
a 3x3 parameter-sensitivity grid, buy-and-hold for comparison, and a verdict:
PASS / INSUFFICIENT_TRADES / FRAGILE / MARGINAL / FAIL_OOS / FAIL.
Only PASS clears the risk manager.

Examples:
  backtest.py --bars trader-workspace/data/NVDA.csv --plan trader-workspace/plans/NVDA-2026-11-06.json \
              --profile trader-workspace/profile.json --out trader-workspace/backtests/NVDA-2026-11-06.json
  backtest.py --bars data/AAPL.csv data/MSFT.csv --setup breakout --stop-atr 2 --rr 2.5 --max-hold 20
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

import traderlib as tl

SETUPS = ("breakout", "pullback", "trend_continuation", "mean_reversion")


@dataclass
class Params:
    setup: str = "breakout"
    stop_atr: Optional[float] = 2.0
    stop_pct: Optional[float] = None
    rr: float = 2.0
    max_hold: int = 20
    lookback: int = 20
    min_relvol: float = 1.0
    capital: float = 10_000.0
    position_value: float = 10_000.0
    commission_per_order: float = 0.0
    commission_per_share: float = 0.0
    fee_bps: float = 0.5
    slippage_bps: float = 5.0


@dataclass
class Trade:
    symbol: str
    entry_time: str
    entry_price: float
    stop: float
    target: float
    qty: float
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    bars_held: int = 0
    fees: float = 0.0
    pnl: float = 0.0
    return_pct: float = 0.0
    r_multiple: float = 0.0
    entry_index: int = 0


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def make_signal(setup: str, f: dict, p: Params) -> Callable[[int], bool]:
    c, h, l, v = f["close"], f["high"], f["low"], f["volume"]
    s50, s200, e20 = f["sma50"], f["sma200"], f["ema20"]
    r14, r2, va = f["rsi14"], f["rsi2"], f["vol_avg20"]

    def trend_ok(i: int) -> bool:
        return (s50[i] is not None and s200[i] is not None
                and c[i] > s50[i] and s50[i] > s200[i])

    if setup == "breakout":
        lb = p.lookback

        def sig(i: int) -> bool:
            if i < lb or not trend_ok(i):
                return False
            prior_high = max(h[i - lb:i])
            if c[i] <= prior_high:
                return False
            if p.min_relvol > 0 and va[i - 1]:
                return v[i] >= p.min_relvol * va[i - 1]
            return True
        return sig

    if setup == "pullback":
        def sig(i: int) -> bool:
            return (s200[i] is not None and s50[i] is not None and e20[i] is not None
                    and r14[i] is not None and c[i] > s200[i] and s50[i] > s200[i]
                    and l[i] <= e20[i] < c[i] and r14[i] <= 60)
        return sig

    if setup == "trend_continuation":
        def sig(i: int) -> bool:
            return (i >= 1 and trend_ok(i) and r14[i] is not None and r14[i - 1] is not None
                    and r14[i - 1] < 50 <= r14[i])
        return sig

    if setup == "mean_reversion":
        def sig(i: int) -> bool:
            return (s200[i] is not None and r2[i] is not None
                    and c[i] > s200[i] and r2[i] < 10)
        return sig

    raise ValueError(f"unknown setup {setup!r}; choose from {', '.join(SETUPS)}")


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(symbol: str, bars: list[dict], p: Params, f: Optional[dict] = None) -> dict:
    f = f or tl.indicator_frame(bars)
    n = len(bars)
    o, h, l, c, t = f["open"], f["high"], f["low"], f["close"], f["time"]
    atr14 = f["atr14"]
    sig = make_signal(p.setup, f, p)
    slip = p.slippage_bps / 1e4
    fee_rate = p.fee_bps / 1e4

    def side_fees(qty: float, price: float) -> float:
        return p.commission_per_order + p.commission_per_share * qty + fee_rate * qty * price

    # first bar where the slowest indicator is defined
    warmup = next((i for i in range(n) if f["sma200"][i] is not None), None)
    if warmup is None:
        warmup = next((i for i in range(n) if f["sma50"][i] is not None), n)

    cash = p.capital
    pos: Optional[Trade] = None
    pending_entry: Optional[int] = None  # signal bar index
    trades: list[Trade] = []
    equity: list[float] = []
    in_market = 0

    for i in range(n):
        # 1) fill a pending entry at this bar's open
        if pending_entry is not None and pos is None:
            si = pending_entry
            pending_entry = None
            fill = o[i] * (1 + slip)
            if p.stop_pct is not None:
                risk_ps = fill * p.stop_pct / 100.0
            else:
                a = atr14[si]
                risk_ps = (p.stop_atr or 2.0) * a if a else None
            if risk_ps and risk_ps > 0 and fill > risk_ps:
                alloc = min(p.position_value, cash)
                qty = alloc / fill if fill > 0 else 0
                if qty > 0:
                    fees = side_fees(qty, fill)
                    cash -= qty * fill + fees
                    pos = Trade(symbol, t[i], fill, fill - risk_ps, fill + p.rr * risk_ps, qty,
                                fees=fees, entry_index=i)

        # 2) manage an open position on this bar
        if pos is not None:
            exit_px = None
            reason = ""
            first_bar = (i == pos.entry_index)
            if not first_bar and o[i] <= pos.stop:
                exit_px, reason = o[i], "gap_stop"
            elif not first_bar and o[i] >= pos.target:
                exit_px, reason = o[i], "gap_target"
            elif l[i] <= pos.stop:
                exit_px, reason = pos.stop, "stop"
            elif h[i] >= pos.target:
                exit_px, reason = pos.target, "target"
            elif i - pos.entry_index + 1 >= p.max_hold:
                exit_px, reason = c[i], "time"
            elif i == n - 1:
                exit_px, reason = c[i], "end_of_data"
            if exit_px is not None:
                fill = exit_px * (1 - slip)
                fees = side_fees(pos.qty, fill)
                cash += pos.qty * fill - fees
                pos.exit_time, pos.exit_price, pos.exit_reason = t[i], fill, reason
                pos.bars_held = i - pos.entry_index + 1
                pos.fees += fees
                cost = pos.qty * pos.entry_price
                pos.pnl = pos.qty * (fill - pos.entry_price) - pos.fees
                pos.return_pct = pos.pnl / cost * 100 if cost else 0.0
                risk = pos.qty * (pos.entry_price - pos.stop)
                pos.r_multiple = pos.pnl / risk if risk > 0 else 0.0
                trades.append(pos)
                pos = None
            else:
                in_market += 1

        equity.append(cash + (pos.qty * c[i] if pos else 0.0))

        # 3) look for a new signal on this close (enter next bar)
        if pos is None and pending_entry is None and i >= warmup and i < n - 1 and sig(i):
            pending_entry = i

    start = warmup if warmup < n else 0
    bh = None
    if start < n and o[start]:
        bh = (c[-1] / o[start] - 1) * 100
    return {
        "symbol": symbol,
        "trades": trades,
        "equity": equity,
        "times": t,
        "start_index": start,
        "in_market_bars": in_market,
        "buy_hold_return_pct": bh,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def trade_stats(trades: list[Trade]) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": None, "avg_win_r": None, "avg_loss_r": None, "payoff": None,
                "expectancy_r": None, "avg_return_pct": None, "profit_factor": None,
                "max_consecutive_losses": 0, "avg_bars_held": None, "total_fees": 0.0, "net_pnl": 0.0}
    wins = [x for x in trades if x.pnl > 0]
    losses = [x for x in trades if x.pnl <= 0]
    avg_win_r = tl.mean(x.r_multiple for x in wins)
    avg_loss_r = tl.mean(x.r_multiple for x in losses)
    payoff = (avg_win_r / abs(avg_loss_r)) if (avg_win_r and avg_loss_r) else None
    gross_win = sum(x.pnl for x in wins)
    gross_loss = -sum(x.pnl for x in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else None)
    streak = best = 0
    for x in trades:
        streak = streak + 1 if x.pnl <= 0 else 0
        best = max(best, streak)
    reasons: dict[str, int] = {}
    for x in trades:
        reasons[x.exit_reason] = reasons.get(x.exit_reason, 0) + 1
    return {
        "trades": n,
        "win_rate": len(wins) / n,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "payoff": payoff,
        "expectancy_r": tl.mean(x.r_multiple for x in trades),
        "avg_return_pct": tl.mean(x.return_pct for x in trades),
        "profit_factor": pf,
        "max_consecutive_losses": best,
        "avg_bars_held": tl.mean(x.bars_held for x in trades),
        "total_fees": sum(x.fees for x in trades),
        "net_pnl": sum(x.pnl for x in trades),
        "exit_reasons": reasons,
    }


def equity_stats(sim: dict, capital: float) -> dict:
    eq = sim["equity"][sim["start_index"]:]
    times = sim["times"][sim["start_index"]:]
    if len(eq) < 2:
        return {}
    peak, mdd = eq[0], 0.0
    for e in eq:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    rets = [(eq[i] / eq[i - 1] - 1) for i in range(1, len(eq)) if eq[i - 1] > 0]
    sd = tl.stdev(rets)
    sharpe = (tl.mean(rets) / sd * math.sqrt(252)) if sd else None
    total = eq[-1] / capital - 1
    d0, d1 = tl.parse_date(times[0]), tl.parse_date(times[-1])
    years = ((d1 - d0).days / 365.25) if (d0 and d1 and d1 > d0) else len(eq) / 252
    cagr = ((eq[-1] / capital) ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else None
    return {
        "start": times[0], "end": times[-1], "years": years,
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100 if cagr is not None else None,
        "max_drawdown_pct": mdd * 100,
        "sharpe": sharpe,
        "exposure_pct": sim["in_market_bars"] / len(eq) * 100,
        "final_equity": eq[-1],
    }


def split_oos(sim: dict, oos_frac: float) -> tuple[list[Trade], list[Trade], str]:
    start = sim["start_index"]
    n = len(sim["times"])
    cut = start + int((n - start) * (1 - oos_frac))
    cut = min(max(cut, start), n - 1)
    is_tr = [x for x in sim["trades"] if x.entry_index < cut]
    oos_tr = [x for x in sim["trades"] if x.entry_index >= cut]
    return is_tr, oos_tr, sim["times"][cut]


def run_all(series: list[tuple[str, list[dict]]], p: Params, frames: Optional[dict] = None) -> list[dict]:
    frames = frames or {}
    return [simulate(sym, bars, p, frames.get(sym)) for sym, bars in series]


def sensitivity(series, base: Params, frames: dict) -> dict:
    grid = []
    stop_key = "stop_pct" if base.stop_pct is not None else "stop_atr"
    base_stop = getattr(base, stop_key) or 2.0
    for sm in (0.75, 1.0, 1.25):
        for rm in (0.75, 1.0, 1.25):
            q = Params(**{**asdict(base), stop_key: round(base_stop * sm, 4), "rr": round(base.rr * rm, 4)})
            sims = run_all(series, q, frames)
            st = trade_stats([x for s in sims for x in s["trades"]])
            grid.append({stop_key: getattr(q, stop_key), "rr": q.rr, "trades": st["trades"],
                         "expectancy_r": st["expectancy_r"], "win_rate": st["win_rate"]})
    pos = sum(1 for g in grid if (g["expectancy_r"] or 0) > 0 and g["trades"] > 0)
    return {"grid": grid, "positive_cells": pos, "cells": len(grid)}


def verdict(all_st: dict, oos_st: dict, sens: Optional[dict], eq: dict, min_trades: int,
            min_expectancy_r: float = 0.1) -> tuple[str, list[str]]:
    notes: list[str] = []
    n = all_st["trades"]
    exp = all_st["expectancy_r"]
    pf = all_st["profit_factor"]
    if n == 0:
        return "FAIL", ["no trades were triggered by this setup in the history provided"]
    if exp is None or exp <= 0 or (pf is not None and pf < 1.0):
        return "FAIL", [f"negative expectancy ({_f(exp)} R per trade, profit factor {_f(pf)}) after costs"]
    if oos_st["trades"] >= 5 and (oos_st["expectancy_r"] or 0) <= 0:
        return "FAIL_OOS", [f"profitable in-sample but out-of-sample expectancy is {_f(oos_st['expectancy_r'])} R "
                            f"over {oos_st['trades']} trades — likely overfit or regime change"]
    if exp < min_expectancy_r:
        return "MARGINAL", [f"expectancy {_f(exp)} R per trade is below the {min_expectancy_r} R minimum — "
                            "too thin to survive real-world execution"]
    if sens and sens["positive_cells"] < math.ceil(sens["cells"] / 2):
        notes.append(f"only {sens['positive_cells']}/{sens['cells']} nearby stop/target settings are profitable")
        return "FRAGILE", notes
    if n < min_trades:
        notes.append(f"only {n} trades (< {min_trades}); add peer symbols with more --bars files or more history")
        return "INSUFFICIENT_TRADES", notes
    if oos_st["trades"] < 5:
        notes.append(f"only {oos_st['trades']} out-of-sample trades; OOS check is weak")
    if eq and (eq.get("max_drawdown_pct") or 0) > 35:
        notes.append(f"max drawdown {eq['max_drawdown_pct']:.1f}% is large")
    return "PASS", notes


def _f(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.2f}"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def params_from_plan(plan: dict, p: Params) -> tuple[Params, list[str]]:
    notes = []
    entry, stop, target = (tl._to_float(plan.get(k)) for k in ("entry", "stop", "target"))
    if plan.get("setup"):
        p.setup = plan["setup"]
    if entry and stop and target and entry > stop:
        risk = entry - stop
        p.rr = round((target - entry) / risk, 4)
        a = tl._to_float(plan.get("atr14") or plan.get("atr"))
        if a:
            p.stop_atr, p.stop_pct = round(risk / a, 4), None
            notes.append(f"plan stop = {p.stop_atr} x ATR14; target = {p.rr} R")
        else:
            p.stop_pct, p.stop_atr = round(risk / entry * 100, 4), None
            notes.append(f"plan stop = {p.stop_pct}% below entry (no ATR in plan); target = {p.rr} R")
    for key in ("max_hold", "lookback", "min_relvol"):
        if plan.get(key) is not None:
            setattr(p, key, type(getattr(p, key))(plan[key]))
    return p, notes


def to_markdown(res: dict) -> str:
    a, io, oo, eq = res["all"], res["in_sample"], res["out_of_sample"], res.get("equity") or {}
    pr = res["params"]
    stop_desc = f"{pr['stop_atr']} x ATR14" if pr.get("stop_atr") else f"{pr['stop_pct']}%"
    lines = [
        f"# Backtest — {', '.join(res['symbols'])} · {pr['setup']}",
        "",
        f"**Verdict: {res['verdict']}**" + (" — " + "; ".join(res["notes"]) if res["notes"] else ""),
        "",
        f"Rule: {pr['setup']} entry next open · stop {stop_desc} · target {pr['rr']} R · "
        f"time stop {pr['max_hold']} bars",
        f"Costs per side: slippage {pr['slippage_bps']} bps, fees {pr['fee_bps']} bps, "
        f"${pr['commission_per_order']}/order, ${pr['commission_per_share']}/share",
        f"Period: {eq.get('start', '?')} → {eq.get('end', '?')} ({_f(eq.get('years'))} yrs)",
        "",
        "| | All | In-sample | Out-of-sample |",
        "|---|---|---|---|",
        f"| Trades | {a['trades']} | {io['trades']} | {oo['trades']} |",
        f"| Win rate | {_pct(a['win_rate'])} | {_pct(io['win_rate'])} | {_pct(oo['win_rate'])} |",
        f"| Expectancy (R) | {_f(a['expectancy_r'])} | {_f(io['expectancy_r'])} | {_f(oo['expectancy_r'])} |",
        f"| Payoff (avg win / avg loss) | {_f(a['payoff'])} | {_f(io['payoff'])} | {_f(oo['payoff'])} |",
        f"| Profit factor | {_f(a['profit_factor'])} | {_f(io['profit_factor'])} | {_f(oo['profit_factor'])} |",
        "",
    ]
    if eq:
        lines += [
            f"Primary symbol equity ({res['symbols'][0]}): total {_f(eq['total_return_pct'])}% · "
            f"CAGR {_f(eq['cagr_pct'])}% · max drawdown {_f(eq['max_drawdown_pct'])}% · "
            f"Sharpe {_f(eq['sharpe'])} · time in market {_f(eq['exposure_pct'])}% · "
            f"buy & hold {_f(res.get('buy_hold_return_pct'))}%",
            f"All symbols, after slippage: fees ${a['total_fees']:.2f}, net P&L ${a['net_pnl']:.2f}",
            "",
        ]
    if res.get("sensitivity"):
        s = res["sensitivity"]
        lines += [f"Sensitivity: {s['positive_cells']}/{s['cells']} neighbouring settings profitable", ""]
    k = res["kelly_inputs"]
    lines += [f"Kelly inputs → win rate {_pct(k['win_rate'])}, payoff {_f(k['payoff'])}, n = {k['trades']}", ""]
    lines.append("_Past performance does not guarantee future results. Backtests ignore taxes, partial fills, "
                 "halts and survivorship bias in the watchlist._")
    return "\n".join(lines) + "\n"


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bars", nargs="+", required=True, help="bar file(s); first is the primary symbol")
    ap.add_argument("--symbol", help="symbol for the first bar file (default: file name)")
    ap.add_argument("--plan", help="plan JSON from the analyst (sets setup, stop, rr, ...)")
    ap.add_argument("--profile", help="profile.json for fees and min trades (optional)")
    ap.add_argument("--setup", choices=SETUPS)
    ap.add_argument("--stop-atr", type=float)
    ap.add_argument("--stop-pct", type=float)
    ap.add_argument("--rr", type=float)
    ap.add_argument("--max-hold", type=int)
    ap.add_argument("--lookback", type=int)
    ap.add_argument("--min-relvol", type=float)
    ap.add_argument("--capital", type=float)
    ap.add_argument("--position-value", type=float)
    ap.add_argument("--commission-per-order", type=float)
    ap.add_argument("--commission-per-share", type=float)
    ap.add_argument("--fee-bps", type=float)
    ap.add_argument("--slippage-bps", type=float)
    ap.add_argument("--start", help="ignore bars before this date (YYYY-MM-DD)")
    ap.add_argument("--end", help="ignore bars after this date (YYYY-MM-DD)")
    ap.add_argument("--oos-frac", type=float, default=0.3, help="fraction of history held out (default 0.3)")
    ap.add_argument("--min-trades", type=int)
    ap.add_argument("--min-expectancy-r", type=float, default=0.1,
                    help="minimum expectancy per trade in R for PASS (default 0.1)")
    ap.add_argument("--no-sensitivity", action="store_true")
    ap.add_argument("--trades", action="store_true", help="include the full trade list in JSON")
    ap.add_argument("--out")
    ap.add_argument("--md")
    args = ap.parse_args(argv)

    p = Params()
    min_trades = 30
    prof = None
    if args.profile or (tl.workspace_dir() / "profile.json").exists():
        prof = tl.load_profile(args.profile, require_valid=False)
        fees = prof.get("fees", {})
        for k in ("commission_per_order", "commission_per_share", "fee_bps", "slippage_bps"):
            if fees.get(k) is not None:
                setattr(p, k, float(fees[k]))
        if isinstance(prof.get("budget"), (int, float)):
            p.capital = float(prof["budget"])
        if isinstance(prof.get("max_position_value"), (int, float)):
            p.position_value = float(prof["max_position_value"])
        min_trades = int(prof.get("backtest_min_trades", 30))

    notes: list[str] = []
    plan = None
    if args.plan:
        plan = tl.load_json(args.plan)
        p, pn = params_from_plan(plan, p)
        notes += pn

    for k in ("setup", "stop_atr", "stop_pct", "rr", "max_hold", "lookback", "min_relvol", "capital",
              "position_value", "commission_per_order", "commission_per_share", "fee_bps", "slippage_bps"):
        v = getattr(args, k)
        if v is not None:
            setattr(p, k, v)
    if args.stop_pct is not None:
        p.stop_atr = None
    elif args.stop_atr is not None:
        p.stop_pct = None
    if args.min_trades is not None:
        min_trades = args.min_trades
    p.position_value = min(p.position_value, p.capital)
    if p.setup not in SETUPS:
        tl.die(f"unknown setup {p.setup!r}")

    series: list[tuple[str, list[dict]]] = []
    for idx, path in enumerate(args.bars):
        sym = (args.symbol if idx == 0 and args.symbol else
               (plan.get("symbol") if idx == 0 and plan and plan.get("symbol") else Path(path).stem.split("_")[0]))
        sym = str(sym).upper()
        bars = tl.load_bars(path, symbol=sym)
        if args.start:
            bars = [b for b in bars if b["time"][:10] >= args.start]
        if args.end:
            bars = [b for b in bars if b["time"][:10] <= args.end]
        if len(bars) < 260:
            notes.append(f"{sym}: only {len(bars)} bars — fetch several years of daily history for a meaningful test")
        series.append((sym, bars))

    frames = {sym: tl.indicator_frame(bars) for sym, bars in series}
    sims = run_all(series, p, frames)
    all_trades = [x for s in sims for x in s["trades"]]
    is_tr, oos_tr = [], []
    oos_start = None
    for s in sims:
        a, b, cut = split_oos(s, args.oos_frac)
        is_tr += a
        oos_tr += b
        if s is sims[0]:
            oos_start = cut
    all_st, is_st, oos_st = trade_stats(all_trades), trade_stats(is_tr), trade_stats(oos_tr)
    eq = equity_stats(sims[0], p.capital)
    sens = None if args.no_sensitivity else sensitivity(series, p, frames)
    v, vnotes = verdict(all_st, oos_st, sens, eq, min_trades, args.min_expectancy_r)

    res = {
        "generated_at": tl.now_iso(),
        "symbols": [s for s, _ in series],
        "params": asdict(p),
        "verdict": v,
        "notes": notes + vnotes,
        "all": all_st,
        "in_sample": is_st,
        "out_of_sample": oos_st,
        "oos_start": oos_start,
        "equity": eq,
        "buy_hold_return_pct": sims[0]["buy_hold_return_pct"],
        "per_symbol": {s["symbol"]: trade_stats(s["trades"]) for s in sims} if len(sims) > 1 else None,
        "sensitivity": sens,
        "kelly_inputs": {"win_rate": all_st["win_rate"], "payoff": all_st["payoff"], "trades": all_st["trades"]},
        "min_trades": min_trades,
    }
    if args.trades:
        res["trade_list"] = [asdict(x) for x in all_trades]
    res = tl.clean_floats(res, 4)
    text = tl.dump_json(res, args.out)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(to_markdown(res), encoding="utf-8")
    if args.out:
        print(f"{v}: {all_st['trades']} trades, win rate {_pct(all_st['win_rate'])}, "
              f"expectancy {_f(all_st['expectancy_r'])} R, PF {_f(all_st['profit_factor'])}, "
              f"OOS expectancy {_f(oos_st['expectancy_r'])} R ({oos_st['trades']} trades)")
        for nte in res["notes"]:
            print(f"  - {nte}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
