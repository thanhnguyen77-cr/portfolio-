#!/usr/bin/env python3
"""Normalise OHLCV bars from any connector into trader-workspace/data/SYMBOL.csv.

Accepts JSON or CSV on stdin or from --in, in any of the common shapes:
  * [{"time": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}, ...]
  * Alpaca style {"bars": [{"t": ..., "o": ..., "h": ..., "l": ..., "c": ..., "v": ...}]}
    or {"bars": {"AAPL": [...]}}
  * TradingView UDF style {"t": [...], "o": [...], "h": [...], "l": [...], "c": [...], "v": [...]}
  * [[time, open, high, low, close, volume], ...]
  * a TradingView "Export chart data" CSV

New bars are merged with any existing file for the symbol (de-duplicated by
timestamp), so history can be fetched in several pages.

  save_bars.py --symbol NVDA --in /tmp/nvda.json
  cat page2.json | save_bars.py --symbol NVDA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import traderlib as tl


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--in", dest="src", help="input file (default: stdin)")
    ap.add_argument("--timeframe", default="1D", help="suffix for non-daily data, e.g. 1h (default 1D)")
    ap.add_argument("--workspace")
    ap.add_argument("--replace", action="store_true", help="overwrite instead of merging")
    args = ap.parse_args(argv)

    sym = args.symbol.split(":")[-1].upper()
    text = Path(args.src).read_text(encoding="utf-8-sig") if args.src else sys.stdin.read()
    fmt = "json" if args.src and args.src.lower().endswith(".json") else None
    try:
        new = tl.parse_bars(text, fmt=fmt, symbol=sym)
    except Exception as e:  # noqa: BLE001
        tl.die(f"could not parse bars: {e}")
    if not new:
        tl.die("no bars found in input")

    name = sym if args.timeframe.upper() == "1D" else f"{sym}_{args.timeframe}"
    out = tl.workspace_dir(args.workspace) / "data" / f"{name}.csv"
    merged = new
    existed = out.exists()
    if existed and not args.replace:
        old = tl.load_bars(out)
        by_time = {b["time"]: b for b in old}
        by_time.update({b["time"]: b for b in new})
        merged = [by_time[k] for k in sorted(by_time)]
    tl.write_bars_csv(merged, out)
    print(f"{sym}: saved {len(merged)} bars ({merged[0]['time']} → {merged[-1]['time']}) to {out}"
          + (f" (merged {len(new)} new rows)" if existed and not args.replace else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
