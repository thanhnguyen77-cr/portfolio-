"""Browser entry point for the Trader desk (runs in Pyodide).

Uses the desk's own scripts unchanged: regime.py, screen.py, backtest.py,
kelly.py, risk_check.py and the replay broker engine.py. The one stand-in is
the analyst: the AI analyst agent runs in Claude Code, so here a rule-based
planner follows the same written rules (ATR- and structure-based stop,
target at the minimum reward:risk, one of the four testable setups).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional

import traderlib as tl
import regime as rg
import screen as sc
import backtest as bt
import kelly as kl
import risk_check as rc

DATA_ROOT = os.environ.get("DESK_DATA_ROOT", "/data")
_cache: dict[tuple, list] = {}


# ------------------------------------------------------------------ data

def dataset_dir(ds: str) -> str:
    return f"{DATA_ROOT}/{ds}"


def symbols(ds: str) -> list[str]:
    d = dataset_dir(ds)
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".csv"))


def all_bars(ds: str, sym: str) -> list[dict]:
    key = (ds, sym)
    if key not in _cache:
        p = f"{dataset_dir(ds)}/{sym}.csv"
        _cache[key] = tl.load_bars(p) if os.path.exists(p) else []
    return _cache[key]


def bars_upto(ds: str, sym: str, asof: str) -> list[dict]:
    bs = all_bars(ds, sym)
    lo, hi = 0, len(bs)
    while lo < hi:
        mid = (lo + hi) // 2
        if bs[mid]["time"][:10] <= asof:
            lo = mid + 1
        else:
            hi = mid
    return bs[:lo]


def calendar(ds: str) -> list[str]:
    """Trading days covered by the stocks (the index files can reach further back)."""
    stocks = [s for s in symbols(ds) if s not in ("SPY", "QQQ", "VIX")]
    ref = max(stocks, key=lambda s: len(all_bars(ds, s))) if stocks else "SPY"
    return [b["time"][:10] for b in all_bars(ds, ref)]


def info(ds: str) -> dict:
    cal = calendar(ds)
    return {"dataset": ds, "symbols": [s for s in symbols(ds) if s not in ("SPY", "QQQ", "VIX")],
            "first": cal[0], "last": cal[-1], "first_tradable": cal[min(len(cal) - 1, 260)]}


# ------------------------------------------------------------------ planner (analyst stand-in)

def make_plan(sym: str, bars: list[dict], setup: str, min_rr: float) -> Optional[dict]:
    f = tl.indicator_frame(bars)
    i = len(bars) - 1
    atr = f["atr14"][i]
    close = f["close"][i]
    if not atr or not close:
        return None
    swing_low = min(f["low"][max(0, i - 9):i + 1])
    if setup == "breakout":
        trigger = round(max(f["high"][max(0, i - 19):i + 1]) + 0.1 * atr, 2)
        entry, entry_limit, entry_type = trigger, round(trigger + 0.25 * atr, 2), "stop_limit"
        why = "Close is pressing the 20-day high in an uptrend, so buy only if price breaks above it"
    else:
        entry, entry_limit, entry_type = round(close, 2), None, "limit"
        setup = "pullback" if setup == "pullback" else "trend_continuation"
        why = ("Uptrend pulling back to its 20-day average" if setup == "pullback"
               else "Established uptrend with momentum, so buy near the last close")
    stop = min(swing_low - 0.1 * atr, entry - 1.5 * atr)
    stop = round(max(stop, entry - 3.0 * atr), 2)
    risk = entry - stop
    target = round(entry + min_rr * risk, 2)
    hi252 = f["high252"][i] if f["high252"] else None
    note = None
    if hi252 and entry < hi252 < target:
        note = f"52-week high {hi252:.2f} sits below the target and may act as resistance"
    return {"symbol": sym, "setup": setup, "as_of": f["time"][i][:10], "entry_type": entry_type,
            "entry": entry, "entry_limit": entry_limit, "stop": stop, "target": target,
            "atr14": round(atr, 4), "stop_atr": round(risk / atr, 2), "rr": min_rr, "max_hold": 20,
            "lookback": 20, "why": why, "note": note,
            "invalidation": [f"daily close below {stop:.2f} before entry"]}


# ------------------------------------------------------------------ one session

def _profile(settings: dict) -> dict:
    p = tl.merge_defaults({
        "budget": float(settings.get("budget", 10000)),
        "cash_reserve": float(settings.get("cash_reserve", 2000)),
        "max_position_value": float(settings.get("max_position", 2500)),
        "max_risk_per_trade_pct": float(settings.get("risk_pct", 1.0)),
        "kelly_fraction": float(settings.get("kelly_fraction", 0.25)),
        "min_reward_risk": float(settings.get("min_rr", 2.0)),
        "max_open_positions": int(settings.get("max_positions", 5)),
        "min_avg_dollar_volume": float(settings.get("min_dollar_volume", 20_000_000)),
    })
    errs = tl.validate_profile(p)
    if errs:
        raise ValueError("; ".join(errs))
    return p


def _regime(ds: str, asof: str) -> dict:
    spy = bars_upto(ds, "SPY", asof)
    qqq = bars_upto(ds, "QQQ", asof) or None
    vix = bars_upto(ds, "VIX", asof) or None
    if len(spy) < 200:
        return {"regime": "caution", "points": 0, "max_points": 0, "size_multiplier": 0.5,
                "details": {}, "note": "not enough index history"}
    return rg.classify(spy, qqq, vix)


def run_session(ds: str, asof: str, settings: dict, account: Optional[dict] = None,
                skip: Optional[set] = None, sensitivity: bool = True, max_candidates: int = 3) -> dict:
    prof = _profile(settings)
    universe = [s for s in settings.get("watchlist") or info(ds)["symbols"] if s not in (skip or set())]
    out: dict[str, Any] = {"asof": asof, "dataset": ds, "steps": []}

    reg = _regime(ds, asof)
    out["regime"] = reg
    out["steps"].append({"agent": "Market regime", "status": reg["regime"],
                         "text": f"{reg['regime'].replace('_', ' ')} ({reg['points']} of {reg['max_points']} points), "
                                 f"size x{reg['size_multiplier']}"})

    snaps = []
    for s in universe:
        bs = bars_upto(ds, s, asof)
        if len(bs) >= 210:
            snaps.append(sc.snapshot_from_bars(s, bs))
    res = sorted((sc.evaluate(x, prof) for x in snaps), key=lambda r: (r["passed"], r["score"]), reverse=True)
    passed = [r for r in res if r["passed"]]
    out["screen"] = tl.clean_floats(res, 3)
    out["steps"].append({"agent": "Screener", "status": f"{len(passed)} of {len(res)} passed",
                         "text": ", ".join(f"{r['symbol']} {r['score']:.0f}" for r in passed[:5]) or "no survivors"})
    out["candidates"] = []
    decision = None
    if reg["regime"] == "risk_off":
        out["decision"] = {"action": "none", "reason": "Market regime is risk off, so no new longs today."}
        return out
    if not passed:
        out["decision"] = {"action": "none", "reason": "No stock passed every screen filter."}
        return out

    peers = [x["symbol"] for x in res if "liquidity" not in x["failed"]][:12]
    for cand in passed[:max_candidates]:
        sym = cand["symbol"]
        bs = bars_upto(ds, sym, asof)
        plan = make_plan(sym, bs, cand["setup"], prof["min_reward_risk"])
        c: dict[str, Any] = {"symbol": sym, "score": cand["score"], "plan": plan}
        out["candidates"].append(c)
        if not plan:
            c["verdict"] = "no plan"
            continue
        # backtest the plan's rule on the candidate plus liquid peers
        p = bt.Params(capital=prof["budget"], position_value=prof["max_position_value"])
        p, _ = bt.params_from_plan(plan, p)
        fees = prof["fees"]
        p.fee_bps, p.slippage_bps = fees["fee_bps"], fees["slippage_bps"]
        series = [(sym, bs)] + [(s, bars_upto(ds, s, asof)) for s in peers if s != sym]
        series = [(s, b) for s, b in series if len(b) >= 260]
        frames = {s: tl.indicator_frame(b) for s, b in series}
        sims = bt.run_all(series, p, frames)
        trades = [x for s in sims for x in s["trades"]]
        is_tr, oos_tr = [], []
        for s in sims:
            a, b, _ = bt.split_oos(s, 0.3)
            is_tr += a
            oos_tr += b
        all_st, oos_st = bt.trade_stats(trades), bt.trade_stats(oos_tr)
        sens = bt.sensitivity(series, p, frames) if sensitivity else None
        verdict, notes = bt.verdict(all_st, oos_st, sens, bt.equity_stats(sims[0], p.capital),
                                    int(prof["backtest_min_trades"]))
        c["backtest"] = tl.clean_floats({"verdict": verdict, "notes": notes, "trades": all_st["trades"],
                                         "win_rate": all_st["win_rate"], "payoff": all_st["payoff"],
                                         "expectancy_r": all_st["expectancy_r"],
                                         "profit_factor": all_st["profit_factor"],
                                         "oos_trades": oos_st["trades"], "oos_expectancy_r": oos_st["expectancy_r"],
                                         "sensitivity": f"{sens['positive_cells']}/{sens['cells']}" if sens else None,
                                         "symbols": [s for s, _ in series]}, 4)
        if verdict != "PASS":
            c["verdict"] = f"backtest {verdict}"
            continue
        worst = plan["entry_limit"] or plan["entry"]
        sz = kl.size(p_hat=all_st["win_rate"], payoff=all_st["payoff"], n_trades=all_st["trades"],
                     entry=worst, stop=plan["stop"], prof=prof, account=account, regime=reg["regime"],
                     target=plan["target"])
        c["sizing"] = tl.clean_floats(sz, 4)
        if sz["decision"] != "SIZE":
            c["verdict"] = "sizing: no trade"
            continue
        order = {"symbol": sym, "side": "buy", "qty": sz["qty"], "order_type": plan["entry_type"],
                 "limit_price": plan["entry_limit"] or plan["entry"],
                 "stop_price": plan["entry"] if plan["entry_type"] == "stop_limit" else None,
                 "stop_loss": plan["stop"], "take_profit": plan["target"]}
        acct = account or {"equity": prof["budget"], "last_equity": prof["budget"], "cash": prof["budget"],
                           "positions": [], "open_orders": []}
        ck = rc.check_entry(order, prof, acct, {"verdict": verdict, "symbols": [s for s, _ in series],
                                                "all": {"trades": all_st["trades"]}, "params": {"max_hold": 20}},
                            sz, None, reg["regime"])
        c["risk"] = {"decision": "APPROVE" if not ck.failed else "REJECT", "checks": ck.items}
        c["order"] = order
        if ck.failed:
            c["verdict"] = "risk check rejected"
            continue
        c["verdict"] = "approved"
        decision = {"action": "buy", "order": order, "symbol": sym}
        break

    for c in out["candidates"]:
        out["steps"].append({"agent": f"Analyst, backtest, sizing: {c['symbol']}", "status": c.get("verdict"),
                             "text": _cand_line(c)})
    out["decision"] = decision or {"action": "none",
                                   "reason": "No candidate cleared every gate: " +
                                             "; ".join(f"{c['symbol']} {c.get('verdict')}" for c in out["candidates"])}
    return out


def _cand_line(c: dict) -> str:
    b = c.get("backtest") or {}
    if not b:
        return c.get("verdict") or ""
    r2 = lambda v: "n/a" if v is None else f"{v:+.2f}"  # noqa: E731
    s = (f"{b['verdict']} over {b['trades']} trades, {r2(b['expectancy_r'])} R per trade, "
         f"{r2(b['oos_expectancy_r'])} R out of sample")
    if c.get("sizing"):
        s += f"; size {c['sizing'].get('qty')} shares ({c['sizing'].get('binding_constraint')})"
    return s


# ------------------------------------------------------------------ walk forward

def simulate(ds: str, start: str, end: str, every: int, settings: dict, progress=None) -> dict:
    """Run the desk every `every` trading days from start to end with a replay broker."""
    os.environ["SIM_DATA"] = dataset_dir(ds)
    os.environ["SIM_STATE"] = "/tmp/sim_state.json"
    import importlib
    import engine as E
    importlib.reload(E)
    E.DATA = __import__("pathlib").Path(dataset_dir(ds))
    E.STATE = __import__("pathlib").Path("/tmp/sim_state.json")
    E._all_bars.cache_clear()
    E.calendar.cache_clear()
    cal = [d for d in calendar(ds) if start <= d <= end]
    if not cal:
        raise ValueError("no trading days in that range")
    budget = float(settings.get("budget", 10000))
    E.init_state(cal[0], budget, {})
    sessions = [cal[k] for k in range(0, len(cal), max(1, int(every)))]
    log = []
    for n, day in enumerate(sessions):
        E.advance(day)
        with E.state(write=False) as st:
            acct = _account_from_state(E, st)
            held = set(st["positions"]) | {o["symbol"] for o in st["orders"]
                                          if o["status"] in ("accepted", "new") and not o.get("parent_id")}
        exits = _time_stops(E, day, int(settings.get("max_hold", 20)))
        r = run_session(ds, day, settings, account=acct, skip=held, sensitivity=False, max_candidates=2)
        placed = None
        if r["decision"]["action"] == "buy":
            o = r["decision"]["order"]
            with E.state() as st:
                try:
                    po = E.place(st, symbol=o["symbol"], side="buy", qty=o["qty"], type=o["order_type"],
                                 time_in_force="day", limit_price=o["limit_price"], stop_price=o["stop_price"],
                                 order_class="bracket", take_profit_limit_price=o["take_profit"],
                                 stop_loss_stop_price=o["stop_loss"])
                    placed = {k: po[k] for k in ("symbol", "qty", "type", "limit_price", "stop_price")}
                except E.SimError as e:
                    placed = {"error": str(e)}
        log.append({"date": day, "regime": r["regime"]["regime"], "passed": sum(1 for x in r["screen"] if x["passed"]),
                    "decision": r["decision"].get("reason") if not placed else None, "placed": placed,
                    "exits": exits, "candidates": [{"symbol": c["symbol"], "verdict": c.get("verdict")}
                                                   for c in r["candidates"]]})
        if progress:
            progress(n + 1, len(sessions), day)
    E.advance(cal[-1])
    with E.state(write=False) as st:
        eq = st["equity_history"]
        fills = st["fills"]
        orders = [E.order_view(st, o) for o in st["orders"] if not o.get("parent_id")]
        positions = dict(st["positions"])
    spy = {b["time"][:10]: b["close"] for b in bars_upto(ds, "SPY", cal[-1])}
    s0 = spy.get(eq[0]["date"])
    curve = [{"date": h["date"], "equity": h["equity"],
              "spy": round(budget * spy[h["date"]] / s0, 2) if s0 and h["date"] in spy else None} for h in eq]
    end_eq = eq[-1]["equity"]
    peak, mdd = 0.0, 0.0
    for h in eq:
        peak = max(peak, h["equity"])
        mdd = max(mdd, (peak - h["equity"]) / peak if peak else 0)
    trades = _round_trips(fills)
    wins = [t for t in trades if t["pnl"] > 0]
    return tl.clean_floats({
        "start": cal[0], "end": cal[-1], "sessions": len(sessions), "curve": curve, "log": log,
        "orders": orders, "fills": fills, "trades": trades, "open_positions": positions,
        "summary": {"start_equity": budget, "end_equity": end_eq, "return_pct": (end_eq / budget - 1) * 100,
                    "spy_return_pct": (curve[-1]["spy"] / budget - 1) * 100 if curve[-1]["spy"] else None,
                    "max_drawdown_pct": mdd * 100, "orders": len(orders),
                    "filled": sum(1 for o in orders if o["status"] == "filled"),
                    "closed_trades": len(trades), "wins": len(wins),
                    "win_rate": len(wins) / len(trades) if trades else None,
                    "net_pnl": sum(t["pnl"] for t in trades)},
    }, 4)


def _account_from_state(E, st: dict) -> dict:
    eq, pv = E.equity(st, st["asof"])
    prev = E.prev_day(st["asof"])
    hist = {h["date"]: h["equity"] for h in st["equity_history"]}
    positions = [{"symbol": s, "qty": p["qty"], "market_value": p["qty"] * (E.close_on(s, st["asof"]) or p["avg_entry_price"])}
                 for s, p in st["positions"].items()]
    open_orders = [{"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                    "limit_price": o.get("limit_price") or o.get("stop_price")}
                   for o in st["orders"] if o["status"] in ("accepted", "new") and not o.get("parent_id")]
    return {"equity": eq, "last_equity": hist.get(prev, eq), "cash": st["cash"], "positions": positions,
            "open_orders": open_orders}


def _time_stops(E, day: str, max_hold: int) -> list:
    out = []
    cal = E.calendar()
    with E.state() as st:
        for sym, p in list(st["positions"].items()):
            opened = p.get("opened")
            if not opened:
                continue
            held = sum(1 for d in cal if opened <= d <= day)
            if held >= max_hold:
                for o in st["orders"]:
                    if o["symbol"] == sym and o["side"] == "sell" and o["status"] in ("new", "held"):
                        o["status"] = "canceled"
                try:
                    E.place(st, symbol=sym, side="sell", qty=p["qty"], type="market", time_in_force="day")
                    out.append({"symbol": sym, "reason": f"time stop after {held} sessions"})
                except E.SimError as e:
                    out.append({"symbol": sym, "error": str(e)})
    return out


def _round_trips(fills: list[dict]) -> list[dict]:
    open_: dict[str, dict] = {}
    trades = []
    for f in fills:
        if f["side"] == "buy":
            open_[f["symbol"]] = f
        elif f["symbol"] in open_:
            e = open_.pop(f["symbol"])
            pnl = (f["price"] - e["price"]) * f["qty"]
            trades.append({"symbol": f["symbol"], "entry_date": e["date"], "entry": e["price"],
                           "exit_date": f["date"], "exit": f["price"], "qty": f["qty"], "pnl": round(pnl, 2),
                           "return_pct": round((f["price"] / e["price"] - 1) * 100, 2), "exit_kind": f["kind"]})
    return trades


def bars_json(ds: str, sym: str, asof: str, n: int = 120) -> str:
    bs = bars_upto(ds, sym, asof)[-n:]
    return json.dumps([[b["time"][:10], b["open"], b["high"], b["low"], b["close"]] for b in bs])


def to_json(obj: Any) -> str:
    return json.dumps(obj, default=str)
