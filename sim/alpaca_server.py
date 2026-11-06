#!/usr/bin/env python3
"""Simulated Alpaca connector (paper) for the historical replay.

Same tool names as the official alpaca-mcp-server v2 that the desk uses.
Account, positions and orders come from the replay broker in engine.py;
market data stops at the replay date.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
from mcp_stdio import Server, ToolError  # noqa: E402

srv = Server("alpaca", instructions=(
    "Alpaca PAPER trading account (historical replay). Market data and the account exist only up to the "
    "replay date's close; orders placed now are eligible from the next trading session."))


def _wrap(fn):
    def inner(**kw):
        try:
            return fn(**kw)
        except E.SimError as e:
            raise ToolError(str(e))
    return inner


def _account(st: dict) -> dict:
    eq, pv = E.equity(st, st["asof"])
    prev = E.prev_day(st["asof"])
    hist = {h["date"]: h["equity"] for h in st["equity_history"]}
    last_eq = hist.get(prev, st["start_cash"]) if prev else st["start_cash"]
    return {"account_number": "SIM-PAPER-001", "status": "ACTIVE", "currency": "USD", "paper": True,
            "cash": round(st["cash"], 2), "buying_power": round(st["cash"] - E._pending_buy_cost(st), 2),
            "equity": round(eq, 2), "last_equity": round(last_eq, 2), "portfolio_value": round(eq, 2),
            "long_market_value": round(pv, 2), "pattern_day_trader": False, "trading_blocked": False,
            "daytrade_count": 0, "as_of": f"{st['asof']} close (historical replay)"}


def _position(st: dict, sym: str, p: dict) -> dict:
    px = E.close_on(sym, st["asof"]) or p["avg_entry_price"]
    prev = E.close_on(sym, E.prev_day(st["asof"]) or st["asof"]) or px
    mv = p["qty"] * px
    cost = p["qty"] * p["avg_entry_price"]
    return {"symbol": sym, "qty": p["qty"], "side": "long", "avg_entry_price": round(p["avg_entry_price"], 4),
            "current_price": px, "lastday_price": prev, "market_value": round(mv, 2), "cost_basis": round(cost, 2),
            "unrealized_pl": round(mv - cost, 2), "unrealized_plpc": round((mv / cost - 1) if cost else 0, 4),
            "opened": p.get("opened")}


@srv.tool("get_account_info", "Get the paper account's balances and status (cash, equity, last_equity, buying power).")
@_wrap
def get_account_info():
    with E.state(write=False) as st:
        return _account(st)


@srv.tool("get_all_positions", "List all open positions.")
@_wrap
def get_all_positions():
    with E.state(write=False) as st:
        return [_position(st, s, p) for s, p in st["positions"].items()]


@srv.tool("get_open_position", "Get the open position for one symbol.", {"symbol": {"type": "string"}}, ["symbol"])
@_wrap
def get_open_position(symbol):
    with E.state(write=False) as st:
        sym = E.norm_symbol(symbol)
        if sym not in st["positions"]:
            raise E.SimError(f"no open position in {sym}")
        return _position(st, sym, st["positions"][sym])


@srv.tool("get_orders", "List orders. status: open | closed | all.",
          {"status": {"type": "string", "enum": ["open", "closed", "all"]}, "limit": {"type": "integer"},
           "symbols": {"type": "string", "description": "comma-separated filter"}})
@_wrap
def get_orders(status="open", limit=50, symbols=None):
    with E.state(write=False) as st:
        want = {E.norm_symbol(s) for s in symbols.split(",")} if symbols else None
        out = []
        for o in reversed(st["orders"]):
            if o.get("parent_id"):
                continue
            is_open = o["status"] in ("new", "accepted", "held") or any(
                E.find(st, lid)["status"] in ("new", "held") for lid in o.get("legs") or [])
            if status == "open" and not is_open or status == "closed" and is_open:
                continue
            if want and o["symbol"] not in want:
                continue
            out.append(E.order_view(st, o))
        return out[:limit]


@srv.tool("get_order_by_id", "Get one order (with bracket legs) by id or client_order_id.",
          {"order_id": {"type": "string"}}, ["order_id"])
@_wrap
def get_order_by_id(order_id):
    with E.state(write=False) as st:
        return E.order_view(st, E.find(st, order_id))


@srv.tool("place_stock_order",
          "Place a stock order. type: market | limit | stop | stop_limit. time_in_force: day | gtc. "
          "order_class: simple | bracket (bracket needs take_profit_limit_price and stop_loss_stop_price). "
          "Whole shares only.",
          {"symbol": {"type": "string"}, "side": {"type": "string", "enum": ["buy", "sell"]},
           "qty": {"type": "number"}, "type": {"type": "string", "enum": ["market", "limit", "stop", "stop_limit"]},
           "time_in_force": {"type": "string", "enum": ["day", "gtc"]},
           "limit_price": {"type": "number"}, "stop_price": {"type": "number"},
           "order_class": {"type": "string", "enum": ["simple", "bracket"]},
           "take_profit_limit_price": {"type": "number"}, "stop_loss_stop_price": {"type": "number"},
           "client_order_id": {"type": "string"}},
          ["symbol", "side", "qty", "type", "time_in_force"])
@_wrap
def place_stock_order(**kw):
    with E.state() as st:
        o = E.place(st, **kw)
        return E.order_view(st, o)


@srv.tool("cancel_order_by_id", "Cancel an open order (cancelling a bracket parent cancels its legs).",
          {"order_id": {"type": "string"}}, ["order_id"])
@_wrap
def cancel_order_by_id(order_id):
    with E.state() as st:
        return E.order_view(st, E.cancel(st, order_id))


@srv.tool("cancel_all_orders", "Cancel all open orders.")
@_wrap
def cancel_all_orders():
    with E.state() as st:
        done = []
        for o in st["orders"]:
            if not o.get("parent_id") and o["status"] in ("new", "accepted"):
                done.append(E.cancel(st, o["id"])["id"])
        return {"canceled": done}


@srv.tool("replace_order_by_id", "Modify an open order's qty, limit_price or stop_price (e.g. move a stop-loss leg).",
          {"order_id": {"type": "string"}, "qty": {"type": "number"}, "limit_price": {"type": "number"},
           "stop_price": {"type": "number"}}, ["order_id"])
@_wrap
def replace_order_by_id(order_id, qty=None, limit_price=None, stop_price=None):
    with E.state() as st:
        return E.order_view(st, E.replace(st, order_id, qty, limit_price, stop_price))


@srv.tool("close_position", "Close (part of) a position at the next open: cancels that symbol's open exit legs, "
          "then sends a market sell.", {"symbol": {"type": "string"}, "qty": {"type": "number"}}, ["symbol"])
@_wrap
def close_position(symbol, qty=None):
    with E.state() as st:
        sym = E.norm_symbol(symbol)
        pos = st["positions"].get(sym)
        if not pos:
            raise E.SimError(f"no open position in {sym}")
        for o in st["orders"]:
            if o["symbol"] == sym and o["side"] == "sell" and o["status"] in ("new", "held"):
                o["status"] = "canceled"
                o["canceled_at"] = st["asof"]
        o = E.place(st, symbol=sym, side="sell", qty=qty or pos["qty"], type="market", time_in_force="day")
        return E.order_view(st, o)


@srv.tool("close_all_positions", "Close every position at the next open.")
@_wrap
def close_all_positions():
    with E.state(write=False) as st:
        syms = list(st["positions"])
    return [close_position(symbol=s) for s in syms]


@srv.tool("get_stock_bars", "Historical bars. timeframe 1Day only in this replay. Dates YYYY-MM-DD.",
          {"symbol": {"type": "string"}, "timeframe": {"type": "string"}, "start": {"type": "string"},
           "end": {"type": "string"}, "limit": {"type": "integer"}}, ["symbol"])
@_wrap
def get_stock_bars(symbol, timeframe="1Day", start=None, end=None, limit=1000):
    with E.state(write=False) as st:
        asof = st["asof"]
    bs = E.bars(symbol, asof)
    if not bs:
        raise E.SimError(f"no data for {symbol}")
    if start:
        bs = [b for b in bs if b["time"] >= start[:10]]
    if end:
        bs = [b for b in bs if b["time"] <= end[:10]]
    bs = bs[-int(limit or 1000):]
    return {"symbol": E.norm_symbol(symbol), "timeframe": "1Day", "count": len(bs),
            "bars": [{"t": b["time"], "o": b["open"], "h": b["high"], "l": b["low"], "c": b["close"],
                      "v": int(b["volume"])} for b in bs]}


@srv.tool("get_stock_latest_quote", "Latest quote (in the replay: the last close).",
          {"symbol": {"type": "string"}}, ["symbol"])
@_wrap
def get_stock_latest_quote(symbol):
    with E.state(write=False) as st:
        px = E.close_on(symbol, st["asof"])
        if px is None:
            raise E.SimError(f"no data for {symbol}")
        return {"symbol": E.norm_symbol(symbol), "bid_price": px, "ask_price": px, "as_of": f"{st['asof']} close"}


@srv.tool("get_news", "News articles for symbols.", {"symbols": {"type": "string"}, "start": {"type": "string"},
                                                       "end": {"type": "string"}, "limit": {"type": "integer"}})
def get_news(symbols=None, start=None, end=None, limit=10):
    return {"news": [], "note": "No historical news feed is available in this replay. Treat news as unknown, "
                                "not as 'no news'."}


@srv.tool("get_clock", "Market clock: current time, open/closed, next open and close.")
@_wrap
def get_clock():
    with E.state(write=False) as st:
        nxt = E.next_day(st["asof"])
        return {"timestamp": f"{st['asof']}T17:30:00-04:00", "is_open": False,
                "next_open": f"{nxt}T09:30:00-04:00" if nxt else None,
                "next_close": f"{nxt}T16:00:00-04:00" if nxt else None}


@srv.tool("get_calendar", "Trading calendar between two dates.",
          {"start": {"type": "string"}, "end": {"type": "string"}})
def get_calendar(start=None, end=None):
    days = [d for d in E.calendar() if (not start or d >= start[:10]) and (not end or d <= end[:10])]
    return [{"date": d, "open": "09:30", "close": "16:00"} for d in days[:400]]


@srv.tool("get_account_activities", "Account activities (fills).", {"activity_types": {"type": "string"},
                                                                     "after": {"type": "string"}})
@_wrap
def get_account_activities(activity_types="FILL", after=None):
    with E.state(write=False) as st:
        return [{"activity_type": "FILL", "transaction_time": f"{f['date']}", "symbol": f["symbol"],
                 "side": f["side"], "qty": f["qty"], "price": f["price"], "order_id": f["order_id"],
                 "leg": f["kind"]} for f in st["fills"] if not after or f["date"] >= after[:10]]


if __name__ == "__main__":
    srv.run()
