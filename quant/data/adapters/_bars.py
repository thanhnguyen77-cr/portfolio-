"""Shared conversion from a vendor OHLCV frame to the canonical bars schema (used by the HTTP adapters)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant.core.calendars import get_calendar
from quant.core.errors import CalendarError
from quant.core.time import now
from quant.core.types import BAR_COLUMNS, AdjustmentState, Instrument
from quant.data.adapters.base import CORPORATE_ACTION_COLUMNS, empty_bars


def empty_actions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CORPORATE_ACTION_COLUMNS})


def session_closes_utc(calendar_code: str, session_dates: pd.Series) -> pd.Series:
    """Map session dates to the calendar's UTC close; NaT where the calendar does not know the date."""
    if session_dates.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    cal = get_calendar(calendar_code)
    start, end = session_dates.min().date(), session_dates.max().date()
    try:
        closes = cal.session_closes(start, end)
    except CalendarError:
        return pd.Series(pd.NaT, index=session_dates.index, dtype="datetime64[ns, UTC]")
    return session_dates.map(closes).astype("datetime64[ns, UTC]")


def canonical_bars(
    inst: Instrument,
    raw: pd.DataFrame,
    vendor: str,
    adjustment_state: AdjustmentState,
    *,
    drop_unfinished: bool = True,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """`raw` has session_date, open, high, low, close, volume in vendor units; returns BAR_COLUMNS rows.

    Prices are multiplied by `inst.price_scale` (pence -> pounds); rows without a close are dropped; the
    current, unfinished session is dropped when `drop_unfinished` and the ingest time is before its close.
    """
    if raw is None or raw.empty:
        return empty_bars()
    df = pd.DataFrame(
        {"session_date": pd.to_datetime(raw["session_date"]).dt.tz_localize(None).dt.normalize()}
    )
    scale = float(inst.price_scale)
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype="float64") * scale
    vol = raw["volume"] if "volume" in raw.columns else 0.0
    df["volume"] = pd.to_numeric(vol, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    df = df[df["close"].notna()].drop_duplicates("session_date", keep="last")
    if start is not None:
        df = df[df["session_date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["session_date"] <= pd.Timestamp(end)]
    if df.empty:
        return empty_bars()
    for col in ("open", "high", "low"):  # fill gaps from the close so OHLC is always complete
        df[col] = df[col].fillna(df["close"])
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["instrument_id"] = inst.instrument_id
    df["currency"] = inst.currency.value
    df["vendor"] = vendor
    df["adjustment_state"] = adjustment_state.value
    df["ts_close_utc"] = session_closes_utc(inst.calendar_code, df["session_date"])
    ingested_at = pd.Timestamp(now())
    df["ingested_at"] = ingested_at
    df = df.sort_values("session_date").reset_index(drop=True)
    if drop_unfinished and not df.empty:
        last_close = df["ts_close_utc"].iloc[-1]
        if pd.notna(last_close) and ingested_at < last_close:
            df = df.iloc[:-1]
    return df.loc[:, list(BAR_COLUMNS)].reset_index(drop=True)


def actions_frame(rows: list[tuple[str, object, str, float, str | None]], vendor: str) -> pd.DataFrame:
    """rows: (instrument_id, ex_date, action_type, value, currency)."""
    if not rows:
        return empty_actions()
    out = pd.DataFrame(rows, columns=["instrument_id", "ex_date", "action_type", "value", "currency"])
    out["ex_date"] = pd.to_datetime(out["ex_date"]).dt.tz_localize(None).dt.normalize()
    out["vendor"] = vendor
    out["recorded_at"] = pd.Timestamp(now())
    return out.loc[:, list(CORPORATE_ACTION_COLUMNS)]
