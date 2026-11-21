#!/usr/bin/env python3
"""Screen a watchlist for the strongest long setups.

Two input modes:

  1. --bars-dir DIR   one OHLCV file per symbol (SYMBOL.csv / SYMBOL.json),
                      as saved from TradingView / Alpaca. Indicators are
                      computed here, so results are reproducible.
  2. --snapshot FILE  a JSON list of per-symbol indicator values that a
                      TradingView screener already computed, e.g.
                      [{"symbol": "AAPL", "close": 190, "sma50": 182,
                        "sma200": 175, "rsi14": 61, "volume": 5.1e7,
                        "avg_volume": 4.8e7, "atr14": 3.2}, ...]

Every symbol is run through hard filters (liquidity, price, trend, RSI band,
volatility). Survivors get a 0-100 score and a setup label
(breakout / pullback / trend_continuation). Output is JSON (stdout or --out)
plus an optional Markdown table (--md).

Example:
  screen.py --bars-dir trader-workspace/data --profile trader-workspace/profile.json \
            --out trader-workspace/screens/2026-11-06.json --md trader-workspace/screens/2026-11-06.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import traderlib as tl

SNAPSHOT_ALIASES = {
    "symbol": ("symbol", "ticker", "name", "s"),
    "close": ("close", "price", "last", "Close"),
    "sma20": ("sma20", "SMA20"),
    "sma50": ("sma50", "SMA50"),
    "sma200": ("sma200", "SMA200"),
    "ema20": ("ema20", "EMA20"),
    "rsi14": ("rsi14", "rsi", "RSI", "RSI14"),
    "atr14": ("atr14", "atr", "ATR"),
    "volume": ("volume", "Volume"),
    "avg_volume": ("avg_volume", "average_volume", "average_volume_30d_calc",
                   "average_volume_10d_calc", "avg_volume_20", "vol_avg20"),
    "high20": ("high20",),
    "high252": ("high252", "price_52_week_high", "52w_high", "High.All"),
    "ret63": ("ret63", "perf_3m", "Perf.3M"),
}


def _snap_get(row: dict, key: str) -> Any:
    for a in SNAPSHOT_ALIASES[key]:
        if a in row and row[a] not in (None, ""):
            return row[a]
    return None


def snapshot_from_bars(symbol: str, bars: list[dict]) -> dict:
    f = tl.indicator_frame(bars)
    i = len(bars) - 1
    c = f["close"]
    sma50 = f["sma50"]
    slope50 = None
    if i >= 10 and sma50[i] is not None and sma50[i - 10] is not None:
        slope50 = tl.pct(sma50[i], sma50[i - 10])
    prev_vol_avg = f["vol_avg20"][i - 1] if i >= 1 else None
    ret63 = tl.pct(c[i], c[i - 63]) if i >= 63 else None
    high20_prev = max(f["high"][max(0, i - 20):i]) if i >= 1 else None
    return {
        "symbol": symbol,
        "time": f["time"][i],
        "bars": len(bars),
        "close": c[i],
        "sma20": f["sma20"][i],
        "sma50": sma50[i],
        "sma200": f["sma200"][i],
        "ema20": f["ema20"][i],
        "sma50_slope10_pct": slope50,
        "rsi14": f["rsi14"][i],
        "atr14": f["atr14"][i],
        "volume": f["volume"][i],
        "avg_volume": prev_vol_avg,
        "high20_prev": high20_prev,
        "high252": f["high252"][i] if f["high252"] else None,
        "ret63": ret63,
    }


def snapshot_from_row(row: dict) -> dict:
    snap = {k: _snap_get(row, k) for k in SNAPSHOT_ALIASES}
    for k, v in list(snap.items()):
        if k != "symbol" and v is not None:
            snap[k] = tl._to_float(v)
    snap["high20_prev"] = snap.pop("high20")
    snap["sma50_slope10_pct"] = tl._to_float(row.get("sma50_slope10_pct"))
    snap["bars"] = None
    snap["time"] = row.get("time")
    return snap


def evaluate(snap: dict, prof: dict) -> dict:
    scfg = prof["screen"]
    close = snap.get("close")
    checks: list[dict] = []

    def check(name: str, ok: Optional[bool], detail: str) -> None:
        checks.append({"name": name, "status": "unknown" if ok is None else ("pass" if ok else "fail"),
                       "detail": detail})

    sma50, sma200 = snap.get("sma50"), snap.get("sma200")
    rsi14, atr14 = snap.get("rsi14"), snap.get("atr14")
    vol, avg_vol = snap.get("volume"), snap.get("avg_volume")

    check("min_price", None if close is None else close >= prof["min_price"],
          f"close {close} vs min {prof['min_price']}")

    dollar_vol = close * avg_vol if (close is not None and avg_vol) else None
    check("liquidity", None if dollar_vol is None else dollar_vol >= prof["min_avg_dollar_volume"],
          f"avg $ volume {dollar_vol:,.0f}" if dollar_vol else "avg volume unavailable")

    if scfg.get("require_close_above_sma50", True):
        check("close_above_sma50", None if (close is None or sma50 is None) else close > sma50,
              f"close {close} vs SMA50 {_fmt(sma50)}")
    if scfg.get("require_sma50_above_sma200", True):
        check("sma50_above_sma200", None if (sma50 is None or sma200 is None) else sma50 > sma200,
              f"SMA50 {_fmt(sma50)} vs SMA200 {_fmt(sma200)}")

    check("rsi_band", None if rsi14 is None else scfg["rsi_min"] <= rsi14 <= scfg["rsi_max"],
          f"RSI14 {_fmt(rsi14)} in [{scfg['rsi_min']}, {scfg['rsi_max']}]")

    rel_vol = vol / avg_vol if (vol is not None and avg_vol) else None
    check("relative_volume", None if rel_vol is None else rel_vol >= scfg["min_relative_volume"],
          f"rel vol {_fmt(rel_vol)} vs min {scfg['min_relative_volume']}")

    atr_pct = atr14 / close * 100 if (atr14 and close) else None
    check("volatility", None if atr_pct is None else atr_pct <= scfg["max_atr_pct"],
          f"ATR% {_fmt(atr_pct)} vs max {scfg['max_atr_pct']}")

    if snap.get("symbol") in set(prof.get("blocklist") or []):
        check("blocklist", False, "symbol is on the profile blocklist")

    failed = [c["name"] for c in checks if c["status"] == "fail"]
    unknown = [c["name"] for c in checks if c["status"] == "unknown"]
    passed = not failed and not unknown

    score = _score(snap, rel_vol, atr_pct) if not failed else 0.0
    setup = _classify(snap, rel_vol)
    return {
        "symbol": snap.get("symbol"),
        "passed": passed,
        "score": round(score, 1),
        "setup": setup,
        "failed": failed,
        "unknown": unknown,
        "metrics": tl.clean_floats({
            "close": close, "sma20": snap.get("sma20"), "sma50": sma50, "sma200": sma200,
            "rsi14": rsi14, "atr14": atr14, "atr_pct": atr_pct, "rel_volume": rel_vol,
            "avg_dollar_volume": dollar_vol, "ret63_pct": snap.get("ret63"),
            "pct_from_52w_high": tl.pct(close, snap.get("high252")),
            "sma50_slope10_pct": snap.get("sma50_slope10_pct"),
            "as_of": snap.get("time"), "bars": snap.get("bars"),
        }, 4),
        "checks": checks,
    }


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _score(snap: dict, rel_vol: Optional[float], atr_pct: Optional[float]) -> float:
    """0-100 composite. Weights: trend 30, momentum 25, RSI 15, volume 15, 52w-high 15."""
    close, sma50, sma200 = snap.get("close"), snap.get("sma50"), snap.get("sma200")
    pts = 0.0
    # Trend (30): distance above SMA50 (sweet spot 0-10%), SMA50 above SMA200, SMA50 rising
    if close and sma50:
        d = (close / sma50 - 1) * 100
        pts += 12 * (_clip(d / 5) if d <= 10 else _clip(1 - (d - 10) / 15))  # too extended fades
    if sma50 and sma200:
        pts += 10 * _clip((sma50 / sma200 - 1) * 100 / 10)
    slope = snap.get("sma50_slope10_pct")
    if slope is not None:
        pts += 8 * _clip(slope / 3)
    elif sma50 and sma200:
        pts += 4
    # Momentum (25): 3-month return
    r63 = snap.get("ret63")
    if r63 is not None:
        pts += 25 * _clip(r63 / 30)
    # RSI (15): best around 55-65
    r = snap.get("rsi14")
    if r is not None:
        pts += 15 * _clip(1 - abs(r - 60) / 15)
    # Volume (15)
    if rel_vol is not None:
        pts += 15 * _clip((rel_vol - 0.8) / 1.2)
    # Near 52-week high (15)
    hi = snap.get("high252")
    if close and hi:
        off = (1 - close / hi) * 100
        pts += 15 * _clip(1 - off / 15)
    # Penalise very high volatility a little
    if atr_pct is not None and atr_pct > 5:
        pts -= min(10, (atr_pct - 5) * 2)
    return max(0.0, min(100.0, pts))


def _classify(snap: dict, rel_vol: Optional[float]) -> str:
    close = snap.get("close")
    hi20 = snap.get("high20_prev")
    rsi14 = snap.get("rsi14")
    ma = snap.get("ema20") or snap.get("sma20")
    if close and hi20 and close >= hi20 * 0.98 and (rel_vol is None or rel_vol >= 1.2):
        return "breakout"
    if close and ma and abs(close / ma - 1) <= 0.03 and (rsi14 is None or rsi14 <= 58):
        return "pullback"
    return "trend_continuation"


def to_markdown(result: dict) -> str:
    lines = [f"# Screen — {result['as_of']}", "",
             f"Universe: {result['universe']} symbols · passed: {result['passed_count']} · "
             f"showing top {len(result['top'])}", "",
             "| # | Symbol | Score | Setup | Close | RSI14 | ATR% | Rel vol | 3m % | From 52w high % |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(result["top"], 1):
        m = r["metrics"]
        lines.append(f"| {n} | {r['symbol']} | {r['score']} | {r['setup']} | {_fmt(m['close'])} | "
                     f"{_fmt(m['rsi14'])} | {_fmt(m['atr_pct'])} | {_fmt(m['rel_volume'])} | "
                     f"{_fmt(m['ret63_pct'])} | {_fmt(m['pct_from_52w_high'])} |")
    rejected = [r for r in result["results"] if not r["passed"]]
    if rejected:
        lines += ["", "## Filtered out", ""]
        for r in rejected:
            why = ", ".join(r["failed"] + [f"{u} (no data)" for u in r["unknown"]])
            lines.append(f"- **{r['symbol']}** — {why}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bars-dir", help="directory of SYMBOL.csv / SYMBOL.json bar files")
    src.add_argument("--snapshot", help="JSON list of per-symbol indicator snapshots")
    ap.add_argument("--symbols", help="comma-separated subset of symbols to screen")
    ap.add_argument("--profile", help="profile.json (default: workspace profile)")
    ap.add_argument("--top", type=int, help="how many to keep (default profile.screen.top_n)")
    ap.add_argument("--min-bars", type=int, default=210, help="minimum bars per symbol (default 210)")
    ap.add_argument("--out", help="write JSON here")
    ap.add_argument("--md", help="write a Markdown summary here")
    args = ap.parse_args(argv)

    prof = tl.load_profile(args.profile)
    wanted = {s.strip().upper() for s in args.symbols.split(",")} if args.symbols else None

    snaps: list[dict] = []
    errors: list[dict] = []
    if args.bars_dir:
        files = sorted(p for p in Path(args.bars_dir).iterdir()
                       if p.suffix.lower() in (".csv", ".json") and not p.name.startswith("."))
        for p in files:
            sym = p.stem.split("_")[0].upper()
            if wanted and sym not in wanted:
                continue
            try:
                bars = tl.load_bars(p, symbol=sym)
            except Exception as e:  # noqa: BLE001 - report and continue
                errors.append({"symbol": sym, "error": str(e)})
                continue
            if len(bars) < args.min_bars:
                errors.append({"symbol": sym, "error": f"only {len(bars)} bars (< {args.min_bars}); "
                               "fetch more history"})
                continue
            snaps.append(snapshot_from_bars(sym, bars))
    else:
        rows = tl.load_json(args.snapshot)
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("results") or list(rows.values())
        for row in rows:
            snap = snapshot_from_row(row)
            if snap["symbol"] is None:
                continue
            snap["symbol"] = str(snap["symbol"]).split(":")[-1].upper()
            if wanted and snap["symbol"] not in wanted:
                continue
            snaps.append(snap)

    if not snaps:
        tl.die("no symbols to screen" + (f" ({len(errors)} errors: {errors})" if errors else ""))

    results = [evaluate(s, prof) for s in snaps]
    results.sort(key=lambda r: (r["passed"], r["score"]), reverse=True)
    top_n = args.top or int(prof["screen"].get("top_n", 5))
    top = [r for r in results if r["passed"]][:top_n]
    as_of = max((s.get("time") or "" for s in snaps), default="") or tl.now_iso()[:10]
    out = {
        "generated_at": tl.now_iso(),
        "as_of": as_of,
        "universe": len(snaps),
        "passed_count": sum(1 for r in results if r["passed"]),
        "top": top,
        "results": results,
        "errors": errors,
        "filters": prof["screen"],
    }
    text = tl.dump_json(out, args.out)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(to_markdown(out), encoding="utf-8")
    if args.out:
        print(f"screened {out['universe']} symbols; {out['passed_count']} passed; "
              f"top: {', '.join(r['symbol'] + ' (' + str(r['score']) + ')' for r in top) or 'none'}")
        if errors:
            print(f"{len(errors)} symbols skipped: " + "; ".join(f"{e['symbol']}: {e['error']}" for e in errors))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
