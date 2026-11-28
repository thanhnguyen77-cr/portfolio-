#!/usr/bin/env python3
"""Read the market regime: risk_on, caution or risk_off.

Transparent points system (max 7 with SPY + QQQ + VIX):
  SPY and QQQ, each: +1 close > SMA50, +1 close > SMA200, +1 SMA50 > SMA200
  VIX:  < 20 → +1 · 20–30 → 0 · > 30 → −2      (skipped if not supplied)
  SPY drawdown from 52-week high > 10% → −1

  ≥ 6 (≥ 5 without VIX)    → risk_on    full size
  ≥ 3                       → caution    half size (position sizer)
  otherwise                 → risk_off   no new longs (risk manager)

  regime.py --spy data/SPY.csv --qqq data/QQQ.csv --vix data/VIX.csv --out regime/2026-11-06.json
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

import traderlib as tl


def index_points(bars: list[dict]) -> tuple[int, dict]:
    f = tl.indicator_frame(bars)
    i = len(bars) - 1
    c, s50, s200 = f["close"][i], f["sma50"][i], f["sma200"][i]
    pts = 0
    detail = {"close": c, "sma50": s50, "sma200": s200, "as_of": f["time"][i]}
    if s50 is not None and c > s50:
        pts += 1
    if s200 is not None and c > s200:
        pts += 1
    if s50 is not None and s200 is not None and s50 > s200:
        pts += 1
    hi = f["high252"][i] if f["high252"] else None
    detail["drawdown_from_52w_high_pct"] = tl.pct(c, hi)
    detail["points"] = pts
    return pts, detail


def classify(spy: list[dict], qqq: Optional[list[dict]] = None, vix: Optional[list[dict]] = None) -> dict:
    total = 0
    details: dict = {}
    p, details["SPY"] = index_points(spy)
    total += p
    if qqq:
        p, details["QQQ"] = index_points(qqq)
        total += p
    dd = details["SPY"]["drawdown_from_52w_high_pct"]
    if dd is not None and dd < -10:
        total -= 1
        details["SPY"]["drawdown_penalty"] = -1
    vix_level = None
    if vix:
        vix_level = vix[-1]["close"]
        vp = 1 if vix_level < 20 else (0 if vix_level <= 30 else -2)
        total += vp
        details["VIX"] = {"close": vix_level, "points": vp, "as_of": vix[-1]["time"]}
    max_pts = 3 + (3 if qqq else 0) + (1 if vix else 0)
    on_thr = 6 if (qqq and vix) else (5 if qqq else 3)
    caution_thr = 3 if qqq else 2
    regime = "risk_on" if total >= on_thr else ("caution" if total >= caution_thr else "risk_off")
    return tl.clean_floats({"regime": regime, "points": total, "max_points": max_pts,
                            "size_multiplier": {"risk_on": 1.0, "caution": 0.5, "risk_off": 0.0}[regime],
                            "details": details, "generated_at": tl.now_iso()}, 4)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spy", required=True, help="SPY (or other broad index ETF) daily bars")
    ap.add_argument("--qqq", help="QQQ daily bars")
    ap.add_argument("--vix", help="VIX daily bars")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = classify(tl.load_bars(args.spy), tl.load_bars(args.qqq) if args.qqq else None,
                   tl.load_bars(args.vix) if args.vix else None)
    text = tl.dump_json(res, args.out)
    if args.out:
        print(f"{res['regime']} ({res['points']}/{res['max_points']} points) → size x{res['size_multiplier']}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
