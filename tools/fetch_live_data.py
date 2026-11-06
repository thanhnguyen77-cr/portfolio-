#!/usr/bin/env python3
"""Fetch recent daily bars for the app's "Latest" data set.

Runs in GitHub Actions (open internet) after the US close. Uses the
open-source yfinance library (github.com/ranaroussi/yfinance), with Stooq's
CSV download as a per-symbol fallback. Prices are split and dividend
adjusted. Writes SYMBOL.csv (time,open,high,low,close,volume) and meta.json.

  python3 tools/fetch_live_data.py --out _live [--years 8]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "JPM", "V", "MA", "UNH", "LLY",
             "COST", "HD", "NFLX", "AMD", "XOM", "BA", "CAT",
             "GLD", "SLV", "USO", "DBC"]   # commodity ETFs: gold, silver, oil, broad basket
INDEXES = {"SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX"}


def drop_unfinished(rows: list[tuple]) -> list[tuple]:
    """Drop today's bar if the US session has not closed yet (it would be a partial day)."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if rows and rows[-1][0] == now.strftime("%Y-%m-%d") and (now.hour, now.minute) < (16, 15):
        return rows[:-1]
    return rows


def write(path: Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([r[0], *(round(float(x), 4) for x in r[1:5]), int(r[5] or 0)])


def via_yfinance(ticker: str, years: int) -> list[tuple]:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=True)
    df = df.dropna(subset=["Close"])
    return [(i.strftime("%Y-%m-%d"), r["Open"], r["High"], r["Low"], r["Close"], r.get("Volume", 0))
            for i, r in df.iterrows()]


def via_stooq(ticker: str) -> list[tuple]:
    sym = {"^VIX": "^vix"}.get(ticker, ticker.lower() + ".us")
    with urllib.request.urlopen(f"https://stooq.com/q/d/l/?s={sym}&i=d", timeout=30) as r:
        text = r.read().decode()
    rows = []
    for d in csv.DictReader(io.StringIO(text)):
        try:
            rows.append((d["Date"], d["Open"], d["High"], d["Low"], d["Close"], d.get("Volume") or 0))
        except KeyError:
            break
    return rows


def intraday(ticker: str) -> list[tuple]:
    """5-minute bars for the last 5 sessions (Yahoo data, delayed)."""
    import yfinance as yf
    df = yf.Ticker(ticker).history(period="5d", interval="5m", auto_adjust=False)
    df = df.dropna(subset=["Close"])
    return [(i.strftime("%Y-%m-%dT%H:%M%z"), r["Open"], r["High"], r["Low"], r["Close"], r.get("Volume", 0))
            for i, r in df.iterrows()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--intraday", action="store_true", help="also fetch 5-minute bars for the last 5 sessions")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done, failed, sources = [], [], {}
    for name, ticker in [(s, s) for s in WATCHLIST] + list(INDEXES.items()):
        rows = []
        for attempt, fn in enumerate((lambda: via_yfinance(ticker, a.years), lambda: via_stooq(ticker))):
            try:
                rows = fn()
                if len(rows) > 250:
                    sources[name] = "yfinance" if attempt == 0 else "stooq"
                    break
            except Exception as e:  # noqa: BLE001
                print(f"{name}: {'yfinance' if attempt == 0 else 'stooq'} failed: {e}", file=sys.stderr)
            time.sleep(1)
        rows = drop_unfinished(rows)
        if len(rows) > 250:
            write(out / f"{name}.csv", rows)
            done.append(name)
            print(f"{name}: {len(rows)} bars to {rows[-1][0]} ({sources[name]})")
        else:
            failed.append(name)
    if "SPY" not in done or len(done) < 10:
        print(f"too little data fetched ({len(done)} symbols); not publishing", file=sys.stderr)
        return 1
    intra_done = []
    if a.intraday:
        (out / "intraday").mkdir(exist_ok=True)
        for name, ticker in [(s, s) for s in WATCHLIST] + [("SPY", "SPY"), ("QQQ", "QQQ")]:
            try:
                rows = intraday(ticker)
                if len(rows) > 20:
                    write(out / "intraday" / f"{name}.csv", rows)
                    intra_done.append(name)
            except Exception as e:  # noqa: BLE001
                print(f"{name}: intraday failed: {e}", file=sys.stderr)
            time.sleep(0.5)
        print(f"intraday: {len(intra_done)} symbols")
    last = max(open(out / f"{s}.csv").read().strip().splitlines()[-1].split(",")[0] for s in done)
    (out / "meta.json").write_text(json.dumps({
        "symbols": done, "failed": failed, "sources": sources, "last_date": last, "intraday": intra_done,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Daily bars, split and dividend adjusted. Intraday 5-minute bars are delayed by 15 minutes or "
                "more and refreshed every 30 minutes during US market hours. Not a live feed."}, indent=1))
    print(f"published {len(done)} symbols, last bar {last}; failed: {failed or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
