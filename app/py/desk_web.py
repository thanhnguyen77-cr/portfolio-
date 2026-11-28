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
        c = evaluate_candidate(ds, asof, cand["symbol"], cand["setup"], prof, reg, peers, account, sensitivity)
        c["score"] = cand["score"]
        out["candidates"].append(c)
        if c.get("verdict") == "approved":
            decision = {"action": "buy", "order": c["order"], "symbol": c["symbol"]}
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


def evaluate_candidate(ds: str, asof: str, sym: str, setup: str, prof: dict, reg: dict, peers: list[str],
                       account: Optional[dict], sensitivity: bool) -> dict:
    """Plan, backtest, size and risk-check one symbol. Returns the candidate record."""
    bs = bars_upto(ds, sym, asof)
    plan = make_plan(sym, bs, setup, prof["min_reward_risk"])
    c: dict[str, Any] = {"symbol": sym, "plan": plan}
    if not plan:
        c["verdict"] = "no plan"
        return c
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
                                     "avg_bars_held": all_st["avg_bars_held"],
                                     "oos_trades": oos_st["trades"], "oos_expectancy_r": oos_st["expectancy_r"],
                                     "sensitivity": f"{sens['positive_cells']}/{sens['cells']}" if sens else None,
                                     "symbols": [s for s, _ in series]}, 4)
    worst = plan["entry_limit"] or plan["entry"]
    # size even when the backtest fails, so the chart tools can show what the rules would allow
    sz = kl.size(p_hat=all_st["win_rate"] or 0.0, payoff=all_st["payoff"] or 1.0, n_trades=all_st["trades"],
                 entry=worst, stop=plan["stop"], prof=prof, account=account, regime=reg["regime"],
                 target=plan["target"])
    c["sizing"] = tl.clean_floats(sz, 4)
    if verdict != "PASS":
        c["verdict"] = f"backtest {verdict}"
        return c
    if sz["decision"] != "SIZE":
        c["verdict"] = "sizing: no trade"
        return c
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
    c["verdict"] = "approved" if not ck.failed else "risk check rejected"
    return c


def analyze_symbol(ds: str, asof: str, sym: str, settings: dict, setup: Optional[str] = None) -> dict:
    """Chart advice for one symbol on one day, whether or not it passes the screen."""
    prof = _profile(settings)
    reg = _regime(ds, asof)
    bs = bars_upto(ds, sym, asof)
    if len(bs) < 210:
        raise ValueError(f"not enough history for {sym} on {asof}")
    snap = sc.snapshot_from_bars(sym, bs)
    screen = sc.evaluate(snap, prof)
    universe = [s for s in settings.get("watchlist") or info(ds)["symbols"]]
    peers = [s for s in universe if s != sym and len(bars_upto(ds, s, asof)) >= 260][:11]
    c = evaluate_candidate(ds, asof, sym, setup or screen["setup"], prof, reg, peers, None, True)
    f = tl.indicator_frame(bs)
    n = 130
    cal = calendar(ds)
    nxt = next((d for d in cal if d > asof), None)
    chart = {"bars": [[b["time"][:10], b["open"], b["high"], b["low"], b["close"], b["volume"]] for b in bs[-n:]],
             "sma20": f["sma20"][-n:], "sma50": f["sma50"][-n:], "sma200": f["sma200"][-n:],
             "high20": f["high20"][-n:], "low20": f["low20"][-n:]}
    return tl.clean_floats({"symbol": sym, "asof": asof, "next_session": nxt, "regime": reg, "screen": screen,
                            "candidate": c, "chart": chart, "note": analyst_note(ds, sym, asof),
                            "profile": {k: prof[k] for k in
                            ("budget", "cash_reserve", "max_position_value", "max_risk_per_trade_pct",
                             "kelly_fraction", "min_reward_risk")}}, 4)


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


# ------------------------------------------------------------------ analyst note

def _ema(xs: list, n: int) -> list:
    return tl.ema(xs, n)


def _pct_rank(xs: list, v: float) -> float:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return 0.5
    k = sum(1 for x in xs if x <= v)
    return k / len(xs)


def _quantile(xs: list, q: float) -> Optional[float]:
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _state_key(f: dict, i: int) -> Optional[tuple]:
    c, s50, s200, r = f["close"][i], f["sma50"][i], f["sma200"][i], f["rsi14"][i]
    if s50 is None or s200 is None or r is None:
        return None
    return (c > s50, c > s200, "low" if r < 40 else ("mid" if r <= 60 else "high"))


def analyst_note(ds: str, sym: str, asof: str, horizon: int = 20) -> dict:
    """A research-desk style read: factors, base rates from the stock's own history, scenarios, levels, cone."""
    bs = bars_upto(ds, sym, asof)
    f = tl.indicator_frame(bs)
    i = len(bs) - 1
    c = f["close"]
    close = c[i]
    atr = f["atr14"][i] or 0.0
    # momentum and volatility
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    macd = [a - b if (a is not None and b is not None) else None for a, b in zip(ema12, ema26)]
    valid = [m for m in macd if m is not None]
    sig = _ema(valid, 9) if len(valid) >= 9 else []
    hist = (macd[i] - sig[-1]) if (macd[i] is not None and sig and sig[-1] is not None) else None
    rets = [c[k] / c[k - 1] - 1 for k in range(max(1, i - 19), i + 1)]
    vol20 = tl.stdev(rets) if len(rets) > 2 else None
    ann_vol = vol20 * math.sqrt(252) * 100 if vol20 else None
    atr_pct = atr / close * 100 if close else None
    atr_hist = [a / cc * 100 for a, cc in zip(f["atr14"][max(0, i - 252):i], c[max(0, i - 252):i]) if a and cc]
    vol_rank = _pct_rank(atr_hist, atr_pct) if atr_pct else 0.5
    sma20, sma50, sma200 = f["sma20"][i], f["sma50"][i], f["sma200"][i]
    slope50 = tl.pct(sma50, f["sma50"][i - 10]) if i >= 10 and sma50 and f["sma50"][i - 10] else None
    slope200 = tl.pct(sma200, f["sma200"][i - 20]) if i >= 20 and sma200 and f["sma200"][i - 20] else None
    ret = {n: tl.pct(close, c[i - n]) if i >= n else None for n in (5, 21, 63, 126, 252)}
    hi252 = max(f["high"][max(0, i - 251):i + 1])
    lo252 = min(f["low"][max(0, i - 251):i + 1])
    pos52 = (close - lo252) / (hi252 - lo252) * 100 if hi252 > lo252 else 50
    # relative strength and beta vs SPY
    spy = bars_upto(ds, "SPY", asof)
    rs63 = beta = None
    if len(spy) > 70:
        sc_ = [b["close"] for b in spy]
        j = len(sc_) - 1
        spy_ret63 = tl.pct(sc_[j], sc_[j - 63])
        rs63 = (ret[63] - spy_ret63) if (ret[63] is not None and spy_ret63 is not None) else None
        n = min(60, i, j)
        xs = [sc_[j - k] / sc_[j - k - 1] - 1 for k in range(n)]
        ys = [c[i - k] / c[i - k - 1] - 1 for k in range(n)]
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else None
    # volume: on-balance volume slope and relative volume
    obv = [0.0]
    for k in range(1, i + 1):
        obv.append(obv[-1] + (f["volume"][k] if c[k] > c[k - 1] else -f["volume"][k] if c[k] < c[k - 1] else 0))
    obv_slope = (obv[i] - obv[i - 20]) / (sum(f["volume"][i - 19:i + 1]) or 1) if i >= 20 else None
    rel_vol = f["volume"][i] / f["vol_avg20"][i - 1] if i >= 1 and f["vol_avg20"][i - 1] else None

    # factor scorecard
    def factor(name, score, reading, why):
        return {"factor": name, "score": score, "reading": reading, "why": why}
    facts = []
    if sma50 and sma200:
        t = 1 if close > sma50 > sma200 else (-1 if close < sma50 < sma200 else 0)
        facts.append(factor("Trend", t, "Bullish" if t > 0 else ("Bearish" if t < 0 else "Mixed"),
                            f"close {close:.2f} vs 50 day {sma50:.2f} and 200 day {sma200:.2f}; 50 day slope {slope50:+.1f}% over 10 sessions" if slope50 is not None else ""))
    r14 = f["rsi14"][i]
    if r14 is not None:
        m = 1 if (r14 > 55 and (hist or 0) > 0) else (-1 if (r14 < 45 and (hist or 0) < 0) else 0)
        facts.append(factor("Momentum", m, "Positive" if m > 0 else ("Negative" if m < 0 else "Neutral"),
                            f"RSI 14 at {r14:.0f}; MACD histogram {'above' if (hist or 0) > 0 else 'below'} zero; 1 month {ret[21]:+.1f}%" if ret[21] is not None else f"RSI 14 at {r14:.0f}"))
    if rs63 is not None:
        s = 1 if rs63 > 3 else (-1 if rs63 < -3 else 0)
        facts.append(factor("Relative strength", s, "Leading" if s > 0 else ("Lagging" if s < 0 else "In line"),
                            f"{rs63:+.1f} points vs SPY over 3 months"))
    if obv_slope is not None:
        v = 1 if obv_slope > 0.1 else (-1 if obv_slope < -0.1 else 0)
        facts.append(factor("Volume", v, "Accumulation" if v > 0 else ("Distribution" if v < 0 else "Balanced"),
                            f"on-balance volume {'rising' if obv_slope > 0 else 'falling'} over 20 sessions; today's volume {rel_vol:.2f}x average" if rel_vol else ""))
    if atr_pct is not None:
        v = -1 if vol_rank > 0.8 else (1 if vol_rank < 0.3 else 0)
        facts.append(factor("Volatility", v, "Elevated" if vol_rank > 0.8 else ("Calm" if vol_rank < 0.3 else "Normal"),
                            f"ATR {atr_pct:.1f}% of price, {vol_rank * 100:.0f}th percentile of its own year; realised vol {ann_vol:.0f}% annualised" if ann_vol else ""))
    p = 1 if pos52 > 80 else (-1 if pos52 < 25 else 0)
    facts.append(factor("52-week range", p, "Near highs" if p > 0 else ("Near lows" if p < 0 else "Mid range"),
                        f"{pos52:.0f}% of the way from the 52-week low {lo252:.2f} to the high {hi252:.2f}"))
    reg = _regime(ds, asof)
    rg = {"risk_on": 1, "caution": 0, "risk_off": -1}[reg["regime"]]
    facts.append(factor("Market", rg, reg["regime"].replace("_", " "), f"{reg['points']} of {reg['max_points']} regime points"))
    total = sum(x["score"] for x in facts)
    bias = "Bullish" if total >= 3 else ("Bearish" if total <= -3 else ("Lean bullish" if total > 0 else ("Lean bearish" if total < 0 else "Neutral")))

    # conditional base rates from this stock's own history
    key = _state_key(f, i)
    cond, uncond = [], []
    for k in range(200, i - horizon):
        fwd = c[k + horizon] / c[k] - 1
        uncond.append(fwd)
        if key and _state_key(f, k) == key:
            cond.append(fwd)
    use = cond if len(cond) >= 30 else uncond
    base = {
        "horizon": horizon, "state": {"above_50": key[0], "above_200": key[1], "rsi": key[2]} if key else None,
        "matching_days": len(cond), "all_days": len(uncond), "used": "matching" if len(cond) >= 30 else "all",
        "p_up": (sum(1 for x in use if x > 0) / len(use)) if use else None,
        "median": _quantile(use, 0.5), "p25": _quantile(use, 0.25), "p75": _quantile(use, 0.75),
        "p10": _quantile(use, 0.1), "p90": _quantile(use, 0.9),
        "p_up_all": (sum(1 for x in uncond if x > 0) / len(uncond)) if uncond else None,
    }
    scen = []
    for name, q, prob in (("Bull", base["p75"], "1 in 4 chance of at least this"), ("Base", base["median"], "the middle outcome"),
                          ("Bear", base["p25"], "1 in 4 chance of this or worse")):
        if q is not None:
            scen.append({"name": name, "return_pct": q * 100, "price": close * (1 + q), "note": prob})
    # projection cone (1 sigma from realised vol) and median path
    cone = []
    if vol20:
        for h in range(1, horizon + 1):
            sd = vol20 * math.sqrt(h)
            med = (base["median"] or 0) * h / horizon
            cone.append({"h": h, "lo": close * (1 + med - sd), "hi": close * (1 + med + sd), "mid": close * (1 + med)})
    # key levels: recent swing points, moving averages, 20 day and 52 week extremes
    levels = []
    for k in range(max(2, i - 60), i - 1):
        if f["high"][k] >= max(f["high"][k - 2:k + 3]):
            levels.append({"price": f["high"][k], "kind": "resistance", "label": f"swing high {f['time'][k][:10]}"})
        if f["low"][k] <= min(f["low"][k - 2:k + 3]):
            levels.append({"price": f["low"][k], "kind": "support", "label": f"swing low {f['time'][k][:10]}"})
    for lab, v in (("20 day average", sma20), ("50 day average", sma50), ("200 day average", sma200),
                   ("52-week high", hi252), ("52-week low", lo252)):
        if v:
            levels.append({"price": v, "kind": "resistance" if v > close else "support", "label": lab})
    above = sorted([l for l in levels if l["price"] > close], key=lambda l: l["price"])[:3]
    below = sorted([l for l in levels if l["price"] <= close], key=lambda l: -l["price"])[:3]
    return tl.clean_floats({
        "symbol": sym, "asof": asof, "close": close, "change_pct": tl.pct(close, c[i - 1]) if i else None,
        "bias": bias, "score": total, "max_score": len(facts), "factors": facts, "base_rates": base, "scenarios": scen,
        "cone": cone, "levels": {"above": above, "below": below},
        "stats": {"atr": atr, "atr_pct": atr_pct, "ann_vol_pct": ann_vol, "beta": beta, "rsi14": r14, "macd_hist": hist,
                  "returns": {str(k): v for k, v in ret.items()}, "pos52": pos52, "rel_vol": rel_vol,
                  "sma20": sma20, "sma50": sma50, "sma200": sma200, "slope50": slope50, "slope200": slope200},
    }, 4)
