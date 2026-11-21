"""Shared helpers for the Trader skill scripts.

Standard library only, so it runs anywhere Python 3.9+ is available
(Claude Code, claude.ai code execution, a laptop).

Covers:
  * loading OHLCV bars from the many shapes a TradingView / Alpaca / CSV
    export can arrive in, normalised to a list of dicts
  * indicators: SMA, EMA, RSI (Wilder), ATR (Wilder), rolling max/min
  * the trader workspace (profile, paths) and small JSON helpers
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

# --------------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------------

WORKSPACE_ENV = "TRADER_WORKSPACE"
DEFAULT_WORKSPACE = "trader-workspace"
SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPTS_DIR.parent


def workspace_dir(explicit: Optional[str] = None) -> Path:
    """Return the trader workspace directory (not created here)."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(WORKSPACE_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / DEFAULT_WORKSPACE).resolve()


def load_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(obj: Any, path: os.PathLike | str | None = None) -> str:
    text = json.dumps(obj, indent=2, sort_keys=False, default=_json_default)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(p)
    return text


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (datetime,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def clean_floats(obj: Any, ndigits: int = 6) -> Any:
    """Round floats and replace NaN/inf with None, recursively."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: clean_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_floats(v, ndigits) for v in obj]
    return obj


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------

PROFILE_DEFAULTS: dict[str, Any] = {
    "version": 1,
    "currency": "USD",
    "budget": None,                  # total capital this desk may use
    "cash_reserve": None,            # cash that must always stay uninvested
    "max_position_value": None,      # max $ in any single stock
    "max_risk_per_trade_pct": 1.0,   # max % of capital lost if a stop is hit
    "kelly_fraction": 0.25,          # fraction of full Kelly actually used
    "max_open_positions": 5,
    "max_total_exposure_pct": 80.0,  # % of budget that may be invested at once
    "max_daily_loss_pct": 3.0,       # stop opening trades after this daily loss
    "min_reward_risk": 2.0,
    "min_price": 5.0,
    "min_avg_dollar_volume": 20_000_000,
    "allow_shorts": False,
    "allow_market_orders": False,
    "require_backtest": True,
    "backtest_min_trades": 30,
    "approval_ttl_minutes": 30,
    "require_human_confirmation": True,
    "mode": "paper",                 # "paper" or "live"
    "live_trading_enabled": False,   # second switch required for live orders
    "blocklist": [],
    "screen": {
        "rsi_min": 45.0,
        "rsi_max": 70.0,
        "require_close_above_sma50": True,
        "require_sma50_above_sma200": True,
        "min_relative_volume": 0.8,
        "max_atr_pct": 8.0,
        "top_n": 5,
    },
    "guard": {
        "require_bracket": True,          # order guard demands a protective stop on entries
        "max_limit_slippage_pct": 0.5,    # submitted limit may differ from the approved one by this much
    },
    "fees": {
        "commission_per_order": 0.0,
        "commission_per_share": 0.0,
        "fee_bps": 0.5,       # regulatory / exchange fees, per side
        "slippage_bps": 5.0,  # assumed slippage, per side
    },
}

REQUIRED_PROFILE_KEYS = ("budget", "cash_reserve", "max_position_value")


def merge_defaults(profile: dict, defaults: dict = PROFILE_DEFAULTS) -> dict:
    out: dict[str, Any] = {}
    for k, v in defaults.items():
        if isinstance(v, dict):
            out[k] = merge_defaults(profile.get(k) or {}, v)
        else:
            out[k] = profile.get(k, v)
    for k, v in profile.items():
        if k not in out:
            out[k] = v
    return out


def validate_profile(p: dict) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    errs: list[str] = []
    for k in REQUIRED_PROFILE_KEYS:
        if not isinstance(p.get(k), (int, float)) or isinstance(p.get(k), bool):
            errs.append(f"{k} must be set to a number")
    if errs:
        return errs
    if p["budget"] <= 0:
        errs.append("budget must be > 0")
    if p["cash_reserve"] < 0:
        errs.append("cash_reserve must be >= 0")
    if p["cash_reserve"] >= p["budget"]:
        errs.append("cash_reserve must be smaller than budget")
    if p["max_position_value"] <= 0:
        errs.append("max_position_value must be > 0")
    if p["max_position_value"] > p["budget"] - p["cash_reserve"]:
        errs.append("max_position_value cannot exceed budget - cash_reserve")
    if not 0 < float(p["max_risk_per_trade_pct"]) <= 5:
        errs.append("max_risk_per_trade_pct must be in (0, 5]")
    if not 0 < float(p["kelly_fraction"]) <= 1:
        errs.append("kelly_fraction must be in (0, 1]")
    if p["mode"] not in ("paper", "live"):
        errs.append("mode must be 'paper' or 'live'")
    if p["mode"] == "live" and not p.get("live_trading_enabled"):
        errs.append("mode is 'live' but live_trading_enabled is false")
    if float(p["min_reward_risk"]) < 1:
        errs.append("min_reward_risk should be >= 1")
    return errs


def load_profile(path: Optional[str] = None, workspace: Optional[str] = None,
                 require_valid: bool = True) -> dict:
    ppath = Path(path) if path else workspace_dir(workspace) / "profile.json"
    if not ppath.exists():
        die(f"profile not found at {ppath}. Run: trader_profile.py init --budget ... "
            "--cash-reserve ... --max-position ...")
    prof = merge_defaults(load_json(ppath))
    if require_valid:
        errs = validate_profile(prof)
        if errs:
            die("invalid profile: " + "; ".join(errs))
    return prof


# --------------------------------------------------------------------------
# Bars
# --------------------------------------------------------------------------

_ALIASES = {
    "time": ("time", "date", "datetime", "timestamp", "t", "Date", "Time"),
    "open": ("open", "o", "Open"),
    "high": ("high", "h", "High"),
    "low": ("low", "l", "Low"),
    "close": ("close", "c", "Close", "adj_close", "Adj Close"),
    "volume": ("volume", "v", "vol", "Volume", "Vol"),
}


def _pick(d: dict, field: str) -> Any:
    lower = {str(k).lower().strip(): v for k, v in d.items()}
    for a in _ALIASES[field]:
        if a in d:
            return d[a]
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def _to_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        v = float(str(x).replace(",", ""))
    except ValueError:
        return None
    if math.isnan(v):
        return None
    return v


def _norm_time(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().lstrip("-").isdigit()):
        v = float(x)
        if v > 1e14:      # microseconds / nanoseconds
            v = v / 1e6 if v < 1e17 else v / 1e9
        elif v > 1e11:    # milliseconds
            v = v / 1000.0
        dt = datetime.fromtimestamp(v, tz=timezone.utc)
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    s = str(x).strip()
    return s


def _rows_from_columnar(d: dict) -> Optional[list[dict]]:
    """TradingView UDF-style {t:[...], o:[...], ...} or {time:[...]}."""
    t = _pick(d, "time")
    c = _pick(d, "close")
    if isinstance(t, list) and isinstance(c, list):
        cols = {f: _pick(d, f) for f in _ALIASES}
        n = len(t)
        rows = []
        for i in range(n):
            rows.append({f: (cols[f][i] if isinstance(cols[f], list) and i < len(cols[f]) else None)
                         for f in cols})
        return rows
    return None


def _extract_rows(obj: Any, symbol: Optional[str] = None) -> list[dict]:
    if isinstance(obj, list):
        if obj and isinstance(obj[0], (list, tuple)):
            # [[time, o, h, l, c, v], ...]
            return [dict(zip(("time", "open", "high", "low", "close", "volume"), r)) for r in obj]
        return obj
    if isinstance(obj, dict):
        col = _rows_from_columnar(obj)
        if col is not None:
            return col
        for key in ("bars", "data", "candles", "results", "values", "ohlcv"):
            if key in obj:
                inner = obj[key]
                if isinstance(inner, dict) and not _rows_from_columnar(inner):
                    # {"bars": {"AAPL": [...]}}
                    if symbol and symbol in inner:
                        return _extract_rows(inner[symbol])
                    if len(inner) == 1:
                        return _extract_rows(next(iter(inner.values())))
                    raise ValueError(f"multiple symbols in file ({', '.join(inner)}); pass --symbol")
                return _extract_rows(inner, symbol)
        if symbol and symbol in obj:
            return _extract_rows(obj[symbol])
    raise ValueError("unrecognised bar data shape")


def parse_bars(text: str, fmt: Optional[str] = None, symbol: Optional[str] = None) -> list[dict]:
    """Parse CSV or JSON text into normalised bars sorted by time."""
    stripped = text.lstrip()
    if fmt is None:
        fmt = "json" if stripped[:1] in "[{" else "csv"
    if fmt == "json":
        rows = _extract_rows(json.loads(text), symbol)
    else:
        rows = list(csv.DictReader(io.StringIO(text)))
    bars: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        b = {
            "time": _norm_time(_pick(r, "time")),
            "open": _to_float(_pick(r, "open")),
            "high": _to_float(_pick(r, "high")),
            "low": _to_float(_pick(r, "low")),
            "close": _to_float(_pick(r, "close")),
            "volume": _to_float(_pick(r, "volume")) or 0.0,
        }
        if b["close"] is None:
            continue
        for f in ("open", "high", "low"):
            if b[f] is None:
                b[f] = b["close"]
        b["high"] = max(b["high"], b["open"], b["close"])
        b["low"] = min(b["low"], b["open"], b["close"])
        bars.append(b)
    if any(b["time"] for b in bars):
        bars.sort(key=lambda b: b["time"])
        dedup: dict[str, dict] = {}
        for b in bars:
            dedup[b["time"]] = b
        bars = list(dedup.values())
    return bars


def load_bars(path: os.PathLike | str, symbol: Optional[str] = None) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    fmt = "json" if p.suffix.lower() == ".json" else None
    bars = parse_bars(text, fmt=fmt, symbol=symbol)
    if not bars:
        raise ValueError(f"no bars parsed from {p}")
    return bars


def write_bars_csv(bars: Sequence[dict], path: os.PathLike | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([b["time"], b["open"], b["high"], b["low"], b["close"], b["volume"]])


def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            if fmt is None:
                dt = datetime.fromisoformat(s)
            else:
                dt = datetime.strptime(s[:10], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------
# Indicators (lists aligned with input; None until enough data)
# --------------------------------------------------------------------------

Series = list  # list[Optional[float]]


def sma(values: Sequence[float], n: int) -> Series:
    out: Series = [None] * len(values)
    if n <= 0:
        return out
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= n:
            s -= values[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def ema(values: Sequence[float], n: int) -> Series:
    out: Series = [None] * len(values)
    if len(values) < n or n <= 0:
        return out
    k = 2.0 / (n + 1)
    prev = sum(values[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(closes: Sequence[float], n: int = 14) -> Series:
    """Wilder's RSI."""
    out: Series = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains = losses = 0.0
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    ag, al = gains / n, losses / n
    out[n] = _rsi_val(ag, al)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0.0)) / n
        al = (al * (n - 1) + max(-d, 0.0)) / n
        out[i] = _rsi_val(ag, al)
    return out


def _rsi_val(ag: float, al: float) -> float:
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    tr = []
    for i in range(len(closes)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            pc = closes[i - 1]
            tr.append(max(highs[i] - lows[i], abs(highs[i] - pc), abs(lows[i] - pc)))
    return tr


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> Series:
    """Wilder's ATR."""
    tr = true_range(highs, lows, closes)
    out: Series = [None] * len(tr)
    if len(tr) < n:
        return out
    prev = sum(tr[:n]) / n
    out[n - 1] = prev
    for i in range(n, len(tr)):
        prev = (prev * (n - 1) + tr[i]) / n
        out[i] = prev
    return out


def rolling_max(values: Sequence[float], n: int) -> Series:
    out: Series = [None] * len(values)
    for i in range(n - 1, len(values)):
        out[i] = max(values[i - n + 1:i + 1])
    return out


def rolling_min(values: Sequence[float], n: int) -> Series:
    out: Series = [None] * len(values)
    for i in range(n - 1, len(values)):
        out[i] = min(values[i - n + 1:i + 1])
    return out


def columns(bars: Sequence[dict]) -> tuple[list, list, list, list, list, list]:
    return ([b["time"] for b in bars], [b["open"] for b in bars], [b["high"] for b in bars],
            [b["low"] for b in bars], [b["close"] for b in bars], [b["volume"] for b in bars])


def indicator_frame(bars: Sequence[dict]) -> dict[str, list]:
    """Compute the standard indicator set used across the desk."""
    t, o, h, l, c, v = columns(bars)
    frame = {
        "time": t, "open": o, "high": h, "low": l, "close": c, "volume": v,
        "sma20": sma(c, 20), "sma50": sma(c, 50), "sma200": sma(c, 200),
        "ema20": ema(c, 20),
        "rsi14": rsi(c, 14), "rsi2": rsi(c, 2),
        "atr14": atr(h, l, c, 14),
        "vol_avg20": sma(v, 20),
        "high20": rolling_max(h, 20), "low20": rolling_min(l, 20),
        "high252": rolling_max(h, min(252, len(h))) if h else [],
    }
    return frame


def pct(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Percent change from b to a."""
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0


def wilson_lower_bound(p_hat: float, n: int, z: float = 1.0) -> float:
    """Lower bound of the Wilson score interval for a proportion."""
    if n <= 0:
        return 0.0
    denom = 1 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom)


def mean(xs: Iterable[float]) -> Optional[float]:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def stdev(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
