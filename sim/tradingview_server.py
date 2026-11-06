#!/usr/bin/env python3
"""Simulated TradingView connector for the historical replay.

Watchlists, quotes with indicator snapshots, OHLCV history, chart-data export
and price alerts. Everything is computed from real daily bars up to the replay
date; alerts are evaluated against later bars as the replay clock advances.
"""

from __future__ import annotations

import csv
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402
from engine import tl  # noqa: E402
from mcp_stdio import Server, ToolError  # noqa: E402

DESK = Path(os.environ.get("SIM_DESK") or os.getcwd()).resolve()
EXCH = {"VIX": "CBOE", "SPY": "AMEX"}

srv = Server("tradingview", instructions=(
    "TradingView (historical replay). Data exists only up to the replay date's close. Symbols may be given "
    "with or without an exchange prefix (NASDAQ:AAPL or AAPL). SPY here is a closing-level total-return index "
    "(1993-02-01 = 100) rebuilt from daily returns, not SPY's quoted price; VIX is CBOE:VIX."))


def _asof() -> str:
    with E.state(write=False) as st:
        return st["asof"]


def _need(sym: str) -> list[dict]:
    bs = E.bars(sym, _asof())
    if not bs:
        raise ToolError(f"symbol {sym} not found")
    return bs


@srv.tool("list_watchlists", "List the user's watchlists.")
def list_watchlists():
    with E.state(write=False) as st:
        return [{"name": k, "count": len(v)} for k, v in st.get("watchlists", {}).items()]


@srv.tool("get_watchlist", "Symbols in a watchlist (with exchange prefixes).", {"name": {"type": "string"}}, ["name"])
def get_watchlist(name):
    with E.state(write=False) as st:
        wl = st.get("watchlists", {})
        if name not in wl:
            raise ToolError(f"watchlist {name!r} not found; have {list(wl)}")
        return {"name": name, "symbols": wl[name]}


@srv.tool("get_quotes",
          "Screener-style snapshot for up to 50 symbols at the last close: close, volume, avg_volume (prior 20 "
          "sessions), sma20, sma50, sma200, ema20, rsi14, atr14, high20 (prior 20-session high), high252, "
          "ret63 (3-month %), sma50_slope10_pct.",
          {"symbols": {"type": "array", "items": {"type": "string"}}}, ["symbols"])
def get_quotes(symbols):
    asof = _asof()
    out = []
    for raw in symbols[:50]:
        sym = E.norm_symbol(raw)
        bs = E.bars(sym, asof)
        if len(bs) < 30:
            out.append({"symbol": sym, "error": "not found or not enough history"})
            continue
        f = tl.indicator_frame(bs)
        i = len(bs) - 1
        s50 = f["sma50"]
        out.append(tl.clean_floats({
            "symbol": sym, "time": f["time"][i], "close": f["close"][i], "open": f["open"][i],
            "high": f["high"][i], "low": f["low"][i], "volume": f["volume"][i],
            "avg_volume": f["vol_avg20"][i - 1], "sma20": f["sma20"][i], "sma50": s50[i], "sma200": f["sma200"][i],
            "ema20": f["ema20"][i], "rsi14": f["rsi14"][i], "atr14": f["atr14"][i],
            "high20": max(f["high"][i - 20:i]), "high252": f["high252"][i],
            "ret63": tl.pct(f["close"][i], f["close"][i - 63]) if i >= 63 else None,
            "sma50_slope10_pct": tl.pct(s50[i], s50[i - 10]) if i >= 10 and s50[i - 10] else None,
        }, 4))
    return {"as_of": asof, "quotes": out}


@srv.tool("get_ohlcv", "Recent daily OHLCV bars as CSV text (max 500). For full history use export_chart_data.",
          {"symbol": {"type": "string"}, "timeframe": {"type": "string"}, "bars": {"type": "integer"}}, ["symbol"])
def get_ohlcv(symbol, timeframe="1D", bars=100):
    bs = _need(symbol)[-min(int(bars or 100), 500):]
    lines = ["time,open,high,low,close,volume"] + [
        f"{b['time']},{b['open']},{b['high']},{b['low']},{b['close']},{int(b['volume'])}" for b in bs]
    return "\n".join(lines)


@srv.tool("export_chart_data",
          "Export a symbol's full daily chart history (like TradingView's 'Export chart data') to a CSV file "
          "inside the current project, with columns time,open,high,low,close,volume. Returns the row count "
          "and date range. Use this for long histories instead of pulling bars into the conversation.",
          {"symbol": {"type": "string"}, "path": {"type": "string", "description": "file path, relative to the project"},
           "timeframe": {"type": "string"}, "bars": {"type": "integer", "description": "0 = all"}},
          ["symbol", "path"])
def export_chart_data(symbol, path, timeframe="1D", bars=0):
    bs = _need(symbol)
    if bars:
        bs = bs[-int(bars):]
    dest = (DESK / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if DESK not in dest.parents and dest != DESK:
        raise ToolError(f"path must be inside the project ({DESK})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bs:
            w.writerow([b["time"], b["open"], b["high"], b["low"], b["close"], int(b["volume"])])
    note = ("closing-level total-return index rebuilt from daily returns (open=high=low=close)"
            if E.norm_symbol(symbol) == "SPY" else "split-adjusted daily OHLCV")
    return {"symbol": E.norm_symbol(symbol), "rows": len(bs), "first": bs[0]["time"], "last": bs[-1]["time"],
            "path": str(dest), "note": note}


@srv.tool("create_alert",
          "Create a price alert with app push notification. condition: crossing | crossing_up | crossing_down | "
          "greater_than | less_than. expiration: YYYY-MM-DD.",
          {"symbol": {"type": "string"}, "condition": {"type": "string"}, "price": {"type": "number"},
           "expiration": {"type": "string"}, "message": {"type": "string"}, "notify_app": {"type": "boolean"}},
          ["symbol", "condition", "price"])
def create_alert(symbol, condition, price, expiration=None, message="", notify_app=True):
    sym = E.norm_symbol(symbol)
    _need(sym)
    with E.state() as st:
        a = {"id": "alrt_" + uuid.uuid4().hex[:8], "symbol": sym, "condition": condition, "price": float(price),
             "expiration": expiration, "message": message, "notify_app": bool(notify_app), "status": "active",
             "created_asof": st["asof"], "triggered_on": None}
        st.setdefault("alerts", []).append(a)
        E.log(st, "alert_created", {k: a[k] for k in ("id", "symbol", "condition", "price", "expiration", "message")})
        return a


@srv.tool("list_alerts", "List alerts and their status (active, triggered, expired, deleted).")
def list_alerts():
    with E.state(write=False) as st:
        return st.get("alerts", [])


@srv.tool("delete_alert", "Delete an alert by id.", {"id": {"type": "string"}}, ["id"])
def delete_alert(id):  # noqa: A002
    with E.state() as st:
        for a in st.get("alerts", []):
            if a["id"] == id:
                a["status"] = "deleted"
                E.log(st, "alert_deleted", {"id": id, "symbol": a["symbol"]})
                return a
    raise ToolError(f"alert {id} not found")


@srv.tool("get_news", "Headlines for a symbol.", {"symbol": {"type": "string"}}, ["symbol"])
def get_news(symbol):
    return {"symbol": E.norm_symbol(symbol), "items": [],
            "note": "No historical news feed is available in this replay. Treat news as unknown, not as 'no news'."}


if __name__ == "__main__":
    srv.run()
