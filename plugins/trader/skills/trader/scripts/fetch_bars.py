#!/usr/bin/env python3
"""Download years of daily bars straight from Alpaca's market-data API.

Price history that comes back from an MCP connector has to pass through the
model's context and be re-typed into a file, which is slow and error-prone for
1,000+ bars. When Alpaca keys are available in the shell environment, this
script fetches the bars directly and writes trader-workspace/data/SYMBOL.csv,
so backtests run on exact data.

Keys (any of): ALPACA_API_KEY + ALPACA_SECRET_KEY, or APCA_API_KEY_ID +
APCA_API_SECRET_KEY. Paper-account keys work for market data. The free plan
serves the IEX feed (--feed iex); a paid data plan can use --feed sip.

  fetch_bars.py NVDA AAPL MSFT --years 5
  fetch_bars.py SPY QQQ --start 2019-01-01 --adjustment all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional

import traderlib as tl

BASE = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"


def keys() -> Optional[tuple[str, str]]:
    k = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    s = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    return (k, s) if k and s else None


def fetch(symbol: str, start: str, end: Optional[str], feed: str, adjustment: str, creds) -> list[dict]:
    bars: list[dict] = []
    token = None
    while True:
        q = {"timeframe": "1Day", "start": start, "limit": 10000, "adjustment": adjustment, "feed": feed}
        if end:
            q["end"] = end
        if token:
            q["page_token"] = token
        url = BASE.format(symbol=urllib.parse.quote(symbol)) + "?" + urllib.parse.urlencode(q)
        req = urllib.request.Request(url, headers={"APCA-API-KEY-ID": creds[0], "APCA-API-SECRET-KEY": creds[1],
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        bars += data.get("bars") or []
        token = data.get("next_page_token")
        if not token:
            return bars


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--start", help="YYYY-MM-DD (overrides --years)")
    ap.add_argument("--end")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip", "delayed_sip", "boats", "otc"])
    ap.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    ap.add_argument("--workspace")
    args = ap.parse_args(argv)

    creds = keys()
    if not creds:
        tl.die("no Alpaca keys in the environment (ALPACA_API_KEY / ALPACA_SECRET_KEY). "
               "Use the TradingView or Alpaca connector tools + save_bars.py instead.", 3)
    start = args.start or (datetime.now(timezone.utc) - timedelta(days=int(args.years * 365.25))).strftime("%Y-%m-%d")
    out_dir = tl.workspace_dir(args.workspace) / "data"
    failed = 0
    for raw in args.symbols:
        sym = raw.split(":")[-1].upper()
        try:
            rows = fetch(sym, start, args.end, args.feed, args.adjustment, creds)
        except urllib.error.HTTPError as e:
            print(f"{sym}: HTTP {e.code} {e.read().decode(errors='replace')[:200]}", file=sys.stderr)
            failed += 1
            continue
        except urllib.error.URLError as e:
            print(f"{sym}: network error {e.reason}", file=sys.stderr)
            failed += 1
            continue
        bars = tl.parse_bars(json.dumps(rows))
        if not bars:
            print(f"{sym}: no bars returned", file=sys.stderr)
            failed += 1
            continue
        tl.write_bars_csv(bars, out_dir / f"{sym}.csv")
        print(f"{sym}: {len(bars)} daily bars {bars[0]['time']} → {bars[-1]['time']} ({args.feed}, {args.adjustment})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
