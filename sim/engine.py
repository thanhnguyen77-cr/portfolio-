"""Historical-replay market + paper broker shared by the simulated connectors.

State lives in one JSON file (SIM_STATE) guarded by a lock, so the Alpaca and
TradingView connector processes and the harness can all touch it.

Time model: the replay clock (`asof`) is a trading date *after its close*.
Data up to and including `asof` exists; nothing later does. Orders placed
during a session become eligible on the next trading day. `advance(to)` walks
the days in between and fills orders against each day's real OHLC bar.

Fill rules (conservative, matching the desk's backtester):
  * limit buy: open if open <= limit, else limit if low <= limit
  * stop / stop-limit buy: triggers when high >= stop; gap above stop fills at
    the open if it is within the limit, otherwise rests as a limit order
  * bracket legs (take-profit limit + stop-loss stop) activate on the entry
    fill and are treated as good-til-cancelled; a gap through a leg fills at
    the open; if one bar touches both, the stop fills first
  * market orders fill at the next open; no commissions
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SIM_DATA") or HERE / "data")
STATE = Path(os.environ.get("SIM_STATE") or HERE / "runs" / "default" / "state.json")

sys.path.insert(0, str(HERE.parent / "plugins" / "trader" / "skills" / "trader" / "scripts"))
import traderlib as tl  # noqa: E402


class SimError(Exception):
    pass


# ---------------------------------------------------------------- data

@lru_cache(maxsize=256)
def _all_bars(sym: str) -> tuple:
    p = DATA / f"{sym}.csv"
    if not p.exists():
        return ()
    out = []
    with open(p, newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({"time": r["time"], "open": float(r["open"]), "high": float(r["high"]),
                        "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])})
    return tuple(out)


def norm_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    return {"^VIX": "VIX", "VIX.X": "VIX", "$VIX": "VIX"}.get(s, s)


def known(sym: str) -> bool:
    return bool(_all_bars(norm_symbol(sym)))


def bars(sym: str, asof: str) -> list[dict]:
    return [b for b in _all_bars(norm_symbol(sym)) if b["time"] <= asof]


def bar_on(sym: str, day: str) -> Optional[dict]:
    for b in _all_bars(norm_symbol(sym)):
        if b["time"] == day:
            return b
    return None


def close_on(sym: str, day: str) -> Optional[float]:
    bs = bars(sym, day)
    return bs[-1]["close"] if bs else None


@lru_cache(maxsize=1)
def calendar() -> tuple:
    return tuple(b["time"] for b in _all_bars("AAPL"))


def next_day(day: str) -> Optional[str]:
    for d in calendar():
        if d > day:
            return d
    return None


def prev_day(day: str) -> Optional[str]:
    prev = None
    for d in calendar():
        if d >= day:
            return prev
        prev = d
    return prev


# ---------------------------------------------------------------- state

@contextmanager
def state(write: bool = True):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(STATE.with_suffix(".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {}
        yield st
        if write:
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1))
            tmp.replace(STATE)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def init_state(asof: str, cash: float, watchlists: dict) -> None:
    with state() as st:
        st.clear()
        st.update({"asof": asof, "start_cash": cash, "cash": cash, "positions": {}, "orders": [],
                   "fills": [], "alerts": [], "watchlists": watchlists, "log": [],
                   "equity_history": [{"date": asof, "cash": cash, "positions_value": 0.0, "equity": cash}]})


def log(st: dict, event: str, detail: Any) -> None:
    st.setdefault("log", []).append({"real_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                     "asof": st.get("asof"), "event": event, "detail": detail})


def equity(st: dict, day: str) -> tuple[float, float]:
    pv = 0.0
    for sym, p in st["positions"].items():
        px = close_on(sym, day) or p["avg_entry_price"]
        pv += p["qty"] * px
    return st["cash"] + pv, pv


# ---------------------------------------------------------------- orders

def _new_order(**kw) -> dict:
    o = {"id": str(uuid.uuid4()), "client_order_id": kw.pop("client_order_id", None) or str(uuid.uuid4())[:12],
         "status": "accepted", "filled_qty": 0, "filled_avg_price": None, "filled_at": None,
         "canceled_at": None, "expired_at": None, "legs": [], "parent_id": None, "triggered": False}
    o.update(kw)
    return o


def _reserved_sell_qty(st: dict, sym: str) -> float:
    return sum(o["qty"] for o in st["orders"]
               if o["symbol"] == sym and o["side"] == "sell" and o["status"] in ("new", "accepted", "held"))


def _pending_buy_cost(st: dict) -> float:
    tot = 0.0
    for o in st["orders"]:
        if o["side"] == "buy" and o["status"] in ("new", "accepted") and not o.get("parent_id"):
            px = o.get("limit_price") or o.get("stop_price") or close_on(o["symbol"], st["asof"]) or 0
            tot += o["qty"] * px
    return tot


def place(st: dict, *, symbol: str, side: str, qty: float, type: str = "market", time_in_force: str = "day",
          limit_price: float | None = None, stop_price: float | None = None, order_class: str = "simple",
          take_profit_limit_price: float | None = None, stop_loss_stop_price: float | None = None,
          client_order_id: str | None = None) -> dict:
    sym = norm_symbol(symbol)
    side = str(side).lower()
    otype = str(type).lower()
    tif = str(time_in_force).lower()
    oclass = str(order_class or "simple").lower()
    if not known(sym) or not bars(sym, st["asof"]):
        raise SimError(f"asset {symbol} not found")
    if side not in ("buy", "sell"):
        raise SimError("side must be buy or sell")
    if qty is None or float(qty) <= 0 or not float(qty).is_integer():
        raise SimError("qty must be a positive whole number of shares")
    qty = int(float(qty))
    if otype not in ("market", "limit", "stop", "stop_limit"):
        raise SimError("type must be market, limit, stop or stop_limit")
    if otype in ("limit", "stop_limit") and not limit_price:
        raise SimError("limit_price is required for limit and stop_limit orders")
    if otype in ("stop", "stop_limit") and not stop_price:
        raise SimError("stop_price is required for stop and stop_limit orders")
    if tif not in ("day", "gtc"):
        raise SimError("time_in_force must be day or gtc")
    if oclass not in ("simple", "bracket"):
        raise SimError("order_class must be simple or bracket in this replay broker")
    last = close_on(sym, st["asof"])
    ref = float(limit_price or stop_price or last)
    if oclass == "bracket":
        if side != "buy":
            raise SimError("bracket orders are supported for buy entries only")
        if not take_profit_limit_price or not stop_loss_stop_price:
            raise SimError("bracket orders need take_profit_limit_price and stop_loss_stop_price")
        if not float(stop_loss_stop_price) < ref < float(take_profit_limit_price):
            raise SimError("bracket must satisfy stop_loss_stop_price < entry < take_profit_limit_price")
    if side == "buy":
        avail = st["cash"] - _pending_buy_cost(st)
        if qty * ref > avail + 1e-6:
            raise SimError(f"insufficient buying power: need {qty * ref:.2f}, available {avail:.2f}")
    else:
        held = st["positions"].get(sym, {}).get("qty", 0)
        free = held - _reserved_sell_qty(st, sym)
        if qty > free:
            raise SimError(f"insufficient qty available for order (held {held}, already reserved by open "
                           f"orders {held - free}); cancel the open exit legs first")
    eligible = next_day(st["asof"])
    if eligible is None:
        raise SimError("market data ends here; no further sessions in this replay")
    parent = _new_order(symbol=sym, side=side, qty=qty, type=otype, time_in_force=tif,
                        limit_price=float(limit_price) if limit_price else None,
                        stop_price=float(stop_price) if stop_price else None, order_class=oclass,
                        submitted_asof=st["asof"], eligible_from=eligible, client_order_id=client_order_id,
                        submitted_real=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    st["orders"].append(parent)
    if oclass == "bracket":
        tp = _new_order(symbol=sym, side="sell", qty=qty, type="limit", time_in_force="gtc",
                        limit_price=float(take_profit_limit_price), status="held", parent_id=parent["id"],
                        leg="take_profit", submitted_asof=st["asof"], eligible_from=eligible)
        sl = _new_order(symbol=sym, side="sell", qty=qty, type="stop", time_in_force="gtc",
                        stop_price=float(stop_loss_stop_price), status="held", parent_id=parent["id"],
                        leg="stop_loss", submitted_asof=st["asof"], eligible_from=eligible)
        st["orders"] += [tp, sl]
        parent["legs"] = [tp["id"], sl["id"]]
    log(st, "order_placed", {k: parent[k] for k in ("id", "symbol", "side", "qty", "type", "limit_price",
                                                     "stop_price", "order_class", "time_in_force")}
        | {"take_profit": take_profit_limit_price, "stop_loss": stop_loss_stop_price})
    return parent


def find(st: dict, oid: str) -> dict:
    for o in st["orders"]:
        if o["id"] == oid or o.get("client_order_id") == oid:
            return o
    raise SimError(f"order {oid} not found")


def cancel(st: dict, oid: str) -> dict:
    o = find(st, oid)
    if o["status"] not in ("new", "accepted", "held"):
        raise SimError(f"order is {o['status']} and cannot be cancelled")
    o["status"] = "canceled"
    o["canceled_at"] = st["asof"]
    for lid in o.get("legs") or []:
        leg = find(st, lid)
        if leg["status"] in ("new", "accepted", "held"):
            leg["status"] = "canceled"
            leg["canceled_at"] = st["asof"]
    log(st, "order_canceled", {"id": o["id"], "symbol": o["symbol"], "leg": o.get("leg")})
    return o


def replace(st: dict, oid: str, qty: float | None = None, limit_price: float | None = None,
            stop_price: float | None = None) -> dict:
    o = find(st, oid)
    if o["status"] not in ("new", "accepted", "held"):
        raise SimError(f"order is {o['status']} and cannot be replaced")
    if qty is not None:
        o["qty"] = int(float(qty))
    if limit_price is not None:
        o["limit_price"] = float(limit_price)
    if stop_price is not None:
        o["stop_price"] = float(stop_price)
    log(st, "order_replaced", {"id": o["id"], "symbol": o["symbol"], "qty": o["qty"],
                               "limit_price": o.get("limit_price"), "stop_price": o.get("stop_price")})
    return o


def _fill(st: dict, o: dict, price: float, day: str, kind: str) -> None:
    price = round(price, 4)
    o["status"] = "filled"
    o["filled_qty"] = o["qty"]
    o["filled_avg_price"] = price
    o["filled_at"] = day
    sym = o["symbol"]
    pos = st["positions"].get(sym)
    if o["side"] == "buy":
        st["cash"] -= o["qty"] * price
        if pos:
            tot = pos["qty"] + o["qty"]
            pos["avg_entry_price"] = (pos["avg_entry_price"] * pos["qty"] + price * o["qty"]) / tot
            pos["qty"] = tot
        else:
            st["positions"][sym] = {"qty": o["qty"], "avg_entry_price": price, "opened": day}
    else:
        st["cash"] += o["qty"] * price
        if pos:
            pos["qty"] -= o["qty"]
            if pos["qty"] <= 0:
                del st["positions"][sym]
    st["fills"].append({"date": day, "order_id": o["id"], "parent_id": o.get("parent_id"), "symbol": sym,
                        "side": o["side"], "qty": o["qty"], "price": price, "kind": kind})
    log(st, "fill", {"date": day, "symbol": sym, "side": o["side"], "qty": o["qty"], "price": price, "kind": kind})


def _try_entry(st: dict, o: dict, b: dict, day: str) -> bool:
    op, hi, lo = b["open"], b["high"], b["low"]
    L, S = o.get("limit_price"), o.get("stop_price")
    t = o["type"]
    if o["side"] == "buy":
        if t == "market":
            _fill(st, o, op, day, "entry"); return True
        if t == "limit" or (t == "stop_limit" and o.get("triggered")):
            if op <= L:
                _fill(st, o, op, day, "entry"); return True
            if lo <= L:
                _fill(st, o, L, day, "entry"); return True
            return False
        if t == "stop":
            if op >= S:
                _fill(st, o, op, day, "entry"); return True
            if hi >= S:
                _fill(st, o, S, day, "entry"); return True
            return False
        if t == "stop_limit":
            if op >= S:
                o["triggered"] = True
                if op <= L:
                    _fill(st, o, op, day, "entry"); return True
                if lo <= L:
                    _fill(st, o, L, day, "entry"); return True
                return False
            if hi >= S:
                o["triggered"] = True
                _fill(st, o, min(S, L), day, "entry"); return True
            return False
    else:  # simple sell (manual exit)
        if t == "market":
            _fill(st, o, op, day, "exit"); return True
        if t == "limit":
            if op >= L:
                _fill(st, o, op, day, "exit"); return True
            if hi >= L:
                _fill(st, o, L, day, "exit"); return True
            return False
        if t in ("stop", "stop_limit"):
            if op <= S:
                _fill(st, o, op, day, "exit"); return True
            if lo <= S:
                _fill(st, o, S, day, "exit"); return True
            return False
    return False


def _manage_legs(st: dict, parent: dict, b: dict, day: str, entry_bar: bool) -> None:
    legs = [find(st, lid) for lid in parent.get("legs") or []]
    tp = next((x for x in legs if x.get("leg") == "take_profit" and x["status"] == "new"), None)
    sl = next((x for x in legs if x.get("leg") == "stop_loss" and x["status"] == "new"), None)
    if not tp or not sl:
        return
    op, hi, lo = b["open"], b["high"], b["low"]
    done = None
    if not entry_bar and op <= sl["stop_price"]:
        _fill(st, sl, op, day, "stop_loss"); done = sl
    elif not entry_bar and op >= tp["limit_price"]:
        _fill(st, tp, op, day, "take_profit"); done = tp
    elif lo <= sl["stop_price"]:
        _fill(st, sl, sl["stop_price"], day, "stop_loss"); done = sl
    elif hi >= tp["limit_price"]:
        _fill(st, tp, tp["limit_price"], day, "take_profit"); done = tp
    if done:
        other = tp if done is sl else sl
        other["status"] = "canceled"
        other["canceled_at"] = day


def _check_alerts(st: dict, day: str) -> None:
    for a in st.get("alerts", []):
        if a["status"] != "active":
            continue
        if a.get("expiration") and day > a["expiration"][:10]:
            a["status"] = "expired"
            continue
        b = bar_on(a["symbol"], day)
        if not b:
            continue
        px, cond = a["price"], a["condition"]
        prev = close_on(a["symbol"], prev_day(day) or day)
        up = prev is not None and prev < px <= b["high"]
        down = prev is not None and prev > px >= b["low"]
        hit = {"crossing_up": up, "crossing_down": down, "crossing": up or down,
               "greater_than": b["high"] >= px, "less_than": b["low"] <= px}.get(cond, up or down)
        if hit:
            a["status"] = "triggered"
            a["triggered_on"] = day
            log(st, "alert_triggered", {"id": a["id"], "symbol": a["symbol"], "price": px, "date": day,
                                        "message": a.get("message")})


def advance(to_day: str) -> dict:
    """Walk the replay clock forward to `to_day`, filling orders on each bar."""
    with state() as st:
        days = [d for d in calendar() if st["asof"] < d <= to_day]
        for day in days:
            for o in list(st["orders"]):
                if o.get("parent_id") or o["status"] not in ("accepted", "new"):
                    continue
                if o["eligible_from"] > day:
                    continue
                o["status"] = "new"
                b = bar_on(o["symbol"], day)
                if not b:
                    continue
                filled = _try_entry(st, o, b, day)
                if filled and o.get("legs"):
                    for lid in o["legs"]:
                        leg = find(st, lid)
                        if leg["status"] == "held":
                            leg["status"] = "new"
                    _manage_legs(st, o, b, day, entry_bar=True)
                elif not filled and o["time_in_force"] == "day":
                    o["status"] = "expired"
                    o["expired_at"] = day
                    for lid in o.get("legs") or []:
                        find(st, lid)["status"] = "canceled"
                    log(st, "order_expired", {"id": o["id"], "symbol": o["symbol"], "date": day})
            for o in list(st["orders"]):
                if o.get("legs") and o["status"] == "filled" and o.get("filled_at") != day:
                    b = bar_on(o["symbol"], day)
                    if b:
                        _manage_legs(st, o, b, day, entry_bar=False)
            _check_alerts(st, day)
            st["asof"] = day
            eq, pv = equity(st, day)
            st["equity_history"].append({"date": day, "cash": round(st["cash"], 2),
                                         "positions_value": round(pv, 2), "equity": round(eq, 2)})
        log(st, "clock_advanced", {"to": st["asof"], "days": len(days)})
        return {"asof": st["asof"], "days": len(days)}


def set_watchlists(wl: dict) -> None:
    with state() as st:
        st["watchlists"] = wl


def order_view(st: dict, o: dict) -> dict:
    v = {k: o.get(k) for k in ("id", "client_order_id", "symbol", "side", "qty", "type", "time_in_force",
                                "limit_price", "stop_price", "order_class", "status", "filled_qty",
                                "filled_avg_price", "filled_at", "canceled_at", "expired_at")}
    v["submitted_at"] = o.get("submitted_asof")
    if o.get("leg"):
        v["leg"] = o["leg"]
    if o.get("legs"):
        v["legs"] = [order_view(st, find(st, lid)) for lid in o["legs"]]
    return v
