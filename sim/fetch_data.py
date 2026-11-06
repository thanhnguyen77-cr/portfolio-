#!/usr/bin/env python3
"""Download the real historical data the replay runs on.

Sources (public, reachable from raw.githubusercontent.com):
  * stocks  - stocknet dataset (Xu & Cohen, ACL 2018): daily Yahoo Finance
              OHLCV for 88 US stocks, 2012-09-04 to 2017-09-01
              https://github.com/yumoxu/stocknet-dataset
  * SPY     - zipline's daily SPY return series (1993-2018); turned into a
              closing-level index (1993-02-01 = 100). It has closes only, and
              its level is an index, not SPY's quoted price.
              https://github.com/quantopian/zipline
  * VIX     - CBOE VIX daily OHLC from the maintained datahub dataset
              https://github.com/datasets/finance-vix

Everything is normalised to time,open,high,low,close,volume in sim/data/.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

STOCKNET = "https://raw.githubusercontent.com/yumoxu/stocknet-dataset/master/price/raw/{sym}.csv"
SPY_RETURNS = "https://raw.githubusercontent.com/quantopian/zipline/master/zipline/resources/market_data/SPY_benchmark.csv"
VIX = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"

STOCKNET_TICKERS = (
    "AAPL ABB ABBV AEP AGFS AMGN AMZN BA BABA BAC BBL BCH BHP BP BRK-A BSAC BUD C CAT CELG CHL CHTR CMCSA "
    "CODI CSCO CVX D DHR DIS DUK EXC FB GD GE GMRE GOOG HD HON HRG HSBC IEP INTC JNJ JPM KO LMT MA MCD MDT "
    "MMM MO MRK MSFT NEE NGG NVS ORCL PCG PCLN PEP PFE PG PICO PM PPL PTR RDS-B REX SLB SNP SNY SO SPLP SRE "
    "T TM TOT TSM UL UN UNH UPS UTX V VZ WFC WMT XOM"
).split()


def get(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8")


def write(path: Path, rows: list[list]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        w.writerows(rows)


def fetch_stock(sym: str) -> int:
    text = get(STOCKNET.format(sym=sym))
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            o, h, l, c = (float(r[k]) for k in ("Open", "High", "Low", "Close"))
            v = float(r["Volume"] or 0)
        except (ValueError, KeyError):
            continue  # stocknet marks missing days with "null"
        rows.append([r["Date"], o, h, l, c, int(v)])
    write(DATA / f"{sym}.csv", rows)
    return len(rows)


def fetch_spy() -> int:
    text = get(SPY_RETURNS)
    level = 100.0
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        level *= 1 + float(r["return"])
        lv = round(level, 4)
        rows.append([r["date"][:10], lv, lv, lv, lv, 0])
    write(DATA / "SPY.csv", rows)
    return len(rows)


def fetch_vix() -> int:
    text = get(VIX)
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        rows.append([r["DATE"], float(r["OPEN"]), float(r["HIGH"]), float(r["LOW"]), float(r["CLOSE"]), 0])
    write(DATA / "VIX.csv", rows)
    return len(rows)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"SPY index: {fetch_spy()} rows")
    print(f"VIX: {fetch_vix()} rows")
    bad = []
    for sym in STOCKNET_TICKERS:
        try:
            n = fetch_stock(sym)
            print(f"{sym}: {n} rows")
        except Exception as e:  # noqa: BLE001
            bad.append(f"{sym} ({e})")
    if bad:
        print("failed: " + ", ".join(bad), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
