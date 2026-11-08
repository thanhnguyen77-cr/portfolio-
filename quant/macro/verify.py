"""Verify the macro catalogue against the live providers: fetch each series over a frequency-appropriate
window and report what came back. This is the only place the `verified` flags should be changed from."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta

import pandas as pd

from quant.core.time import today
from quant.data.adapters.base import Capability, MarketDataAdapter
from quant.macro.catalog import MACRO_SERIES, MacroSeries

LOOKBACK_DAYS = {"daily": 60, "monthly": 240, "quarterly": 400}


def verify_catalog(
    adapters: Mapping[str, MarketDataAdapter],
    *,
    series: Sequence[MacroSeries] | None = None,
    end: date | None = None,
    min_obs: int = 2,
) -> pd.DataFrame:
    """One row per catalogued series: status ok | fail | skipped (no adapter for the vendor), observations,
    last date and the provider's error text. Nothing is written back; flip `verified` by hand from this."""
    end_d = end or today() - timedelta(days=3)
    rows = []
    for spec in series if series is not None else MACRO_SERIES.values():
        adapter = adapters.get(spec.vendor)
        row = {
            "name": spec.name,
            "vendor": spec.vendor,
            "series_id": spec.series_id,
            "frequency": spec.frequency,
            "verified_flag": spec.verified,
            "status": "skipped",
            "n_obs": 0,
            "last_date": None,
            "error": "",
        }
        if adapter is None or Capability.MACRO not in adapter.capabilities():
            row["error"] = f"no configured adapter for vendor {spec.vendor!r}"
            rows.append(row)
            continue
        start_d = end_d - timedelta(days=LOOKBACK_DAYS.get(spec.frequency, 150))
        try:
            res = adapter.get_macro_series([spec.series_id], start_d, end_d)
        except Exception as exc:  # noqa: BLE001 - report, never abort the sweep
            row["status"], row["error"] = "fail", f"{type(exc).__name__}: {str(exc)[:160]}"
            rows.append(row)
            continue
        err = res.errors.get(spec.series_id)
        frame = res.frame[res.frame["series_id"] == spec.series_id] if not res.frame.empty else res.frame
        row["n_obs"] = int(len(frame))
        row["last_date"] = frame["date"].max().date() if len(frame) else None
        if err or len(frame) < min_obs:
            row["status"] = "fail"
            row["error"] = err or f"only {len(frame)} observation(s) in {start_d}..{end_d}"
        else:
            row["status"] = "ok"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> str:
    counts = table["status"].value_counts().to_dict()
    flips = table[(table["status"] == "ok") & (~table["verified_flag"])]["name"].tolist()
    stale = table[(table["status"] == "fail") & (table["verified_flag"])]["name"].tolist()
    lines = [f"catalogue check: {counts}"]
    if flips:
        lines.append(f"  verified live but flagged unverified (flip to True): {flips}")
    if stale:
        lines.append(f"  flagged verified but FAILED live (investigate): {stale}")
    return "\n".join(lines)
