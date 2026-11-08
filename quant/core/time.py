"""Time utilities and the test seam for "now".

All timestamps inside quant are timezone-aware UTC; naive inputs are treated as UTC. `now()` is the only
clock the package uses, so tests freeze it with `frozen_now` instead of monkeypatching `datetime`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone

import pandas as pd

UTC = timezone.utc

_now_override: Callable[[], datetime] | None = None


def _real_now() -> datetime:
    return datetime.now(tz=UTC)


def now() -> datetime:
    """Current UTC time (tz-aware). Overridable in tests via `frozen_now`."""
    fn = _now_override or _real_now
    return ensure_utc(fn())


def set_now(fn: Callable[[], datetime] | None) -> None:
    """Install (or clear, with None) a replacement clock."""
    global _now_override
    _now_override = fn


@contextmanager
def frozen_now(at: datetime | str | pd.Timestamp) -> Iterator[datetime]:
    """Freeze `now()` at `at` for the duration of the block."""
    fixed = ensure_utc(pd.Timestamp(at).to_pydatetime())
    previous = _now_override
    set_now(lambda: fixed)
    try:
        yield fixed
    finally:
        set_now(previous)


def ensure_utc(ts: datetime | pd.Timestamp) -> datetime | pd.Timestamp:
    """Return `ts` as tz-aware UTC. Naive values are assumed to already be UTC."""
    if isinstance(ts, pd.Timestamp):
        return ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    raise TypeError(f"expected datetime or Timestamp, got {type(ts).__name__}")


def to_date(value: str | date | datetime | pd.Timestamp) -> date:
    """Coerce common date-like inputs to a calendar `date` without shifting time zones."""
    if isinstance(value, pd.Timestamp | datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return pd.Timestamp(value).date()
    raise TypeError(f"cannot convert {type(value).__name__} to date")


def today() -> date:
    return now().date()
