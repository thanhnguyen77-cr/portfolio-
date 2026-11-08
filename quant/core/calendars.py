"""Trading calendars: exchange calendars via `exchange_calendars`, plus FX and crypto proxies.

Session dates are tz-naive calendar dates; opens and closes are tz-aware UTC timestamps. All lookups raise
`CalendarError` for unknown codes or out-of-bounds dates rather than silently returning something.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from functools import cache, reduce
from zoneinfo import ZoneInfo

import pandas as pd

from quant.core.errors import CalendarError
from quant.core.time import to_date

CALENDAR_START = "1990-01-01"
# Codes exchange_calendars ships. CME Group venues (CBOT, COMEX, NYMEX) trade on Globex and map to CMES.
EXCHANGE_CODES: frozenset[str] = frozenset({"XNYS", "XNAS", "XLON", "CMES"})
SYNTHETIC_CODES: frozenset[str] = frozenset({"FX", "CRYPTO"})

DateLike = str | date | datetime | pd.Timestamp


class TradingCalendar:
    """Common interface. Subclasses implement `is_session`, `session_open`, `session_close`, `sessions`."""

    code: str = ""

    def sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        raise NotImplementedError

    def is_session(self, d: DateLike) -> bool:
        raise NotImplementedError

    def session_open(self, d: DateLike) -> pd.Timestamp:
        raise NotImplementedError

    def session_close(self, d: DateLike) -> pd.Timestamp:
        raise NotImplementedError

    def date_to_session(self, d: DateLike, direction: str = "previous") -> date:
        """Nearest session on or before (`previous`) / on or after (`next`) `d`."""
        step = 1 if direction == "next" else -1
        if direction not in {"next", "previous"}:
            raise ValueError("direction must be 'next' or 'previous'")
        current = to_date(d)
        for _ in range(60):
            if self.is_session(current):
                return current
            current += timedelta(days=step)
        raise CalendarError(f"{self.code}: no session within 60 days of {d}")

    def next_session(self, d: DateLike) -> date:
        """First session strictly after `d`."""
        return self.date_to_session(to_date(d) + timedelta(days=1), "next")

    def previous_session(self, d: DateLike) -> date:
        """Last session strictly before `d`."""
        return self.date_to_session(to_date(d) - timedelta(days=1), "previous")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.code!r})"


class ExchangeCalendarAdapter(TradingCalendar):
    """Thin wrapper over an `exchange_calendars` calendar (holidays, early closes, DST handled upstream)."""

    def __init__(self, code: str) -> None:
        import exchange_calendars as xcals

        self.code = code
        try:
            self._cal = xcals.get_calendar(code, start=CALENDAR_START, side="left")
        except Exception as exc:  # noqa: BLE001 - upstream raises several ad-hoc error types
            raise CalendarError(f"cannot build exchange calendar {code!r}: {exc}") from exc

    def _ts(self, d: DateLike) -> pd.Timestamp:
        ts = pd.Timestamp(to_date(d))
        if ts < self._cal.first_session or ts > self._cal.last_session:
            raise CalendarError(
                f"{self.code}: {ts.date()} is outside calendar bounds "
                f"{self._cal.first_session.date()}..{self._cal.last_session.date()}"
            )
        return ts

    def sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        return self._cal.sessions_in_range(self._ts(start), self._ts(end))

    def is_session(self, d: DateLike) -> bool:
        return bool(self._cal.is_session(self._ts(d)))

    def session_open(self, d: DateLike) -> pd.Timestamp:
        return self._cal.session_open(self._session(d)).tz_convert("UTC")

    def session_close(self, d: DateLike) -> pd.Timestamp:
        return self._cal.session_close(self._session(d)).tz_convert("UTC")

    def date_to_session(self, d: DateLike, direction: str = "previous") -> date:
        if direction not in {"next", "previous"}:
            raise ValueError("direction must be 'next' or 'previous'")
        return self._cal.date_to_session(self._ts(d), direction=direction).date()

    def _session(self, d: DateLike) -> pd.Timestamp:
        ts = self._ts(d)
        if not self._cal.is_session(ts):
            raise CalendarError(f"{self.code}: {ts.date()} is not a trading session")
        return ts


class WeekdayFXCalendar(TradingCalendar):
    """Spot-FX proxy: every weekday is a session, closing 17:00 New York (so 21:00/22:00 UTC by DST).

    FX trades on most public holidays, so none are excluded. Vendor daily FX bars (e.g. Yahoo) are stamped
    on the New York date, which this calendar matches.
    """

    code = "FX"
    _ny = ZoneInfo("America/New_York")

    def sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        return pd.bdate_range(pd.Timestamp(to_date(start)), pd.Timestamp(to_date(end)))

    def is_session(self, d: DateLike) -> bool:
        return to_date(d).weekday() < 5

    def session_close(self, d: DateLike) -> pd.Timestamp:
        day = to_date(d)
        if not self.is_session(day):
            raise CalendarError(f"FX: {day} is a weekend")
        return pd.Timestamp(datetime.combine(day, time(17, 0), tzinfo=self._ny)).tz_convert("UTC")

    def session_open(self, d: DateLike) -> pd.Timestamp:
        return self.session_close(d) - pd.Timedelta(days=1)


class AllDaysCryptoCalendar(TradingCalendar):
    """24/7 proxy: every calendar day is a session running 00:00:00-23:59:59 UTC."""

    code = "CRYPTO"

    def sessions(self, start: DateLike, end: DateLike) -> pd.DatetimeIndex:
        return pd.date_range(pd.Timestamp(to_date(start)), pd.Timestamp(to_date(end)), freq="D")

    def is_session(self, d: DateLike) -> bool:
        to_date(d)
        return True

    def session_open(self, d: DateLike) -> pd.Timestamp:
        return pd.Timestamp(to_date(d), tz="UTC")

    def session_close(self, d: DateLike) -> pd.Timestamp:
        return pd.Timestamp(to_date(d), tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)


@cache
def _build_calendar(code: str) -> TradingCalendar:
    if code == "FX":
        return WeekdayFXCalendar()
    if code == "CRYPTO":
        return AllDaysCryptoCalendar()
    if code in EXCHANGE_CODES:
        return ExchangeCalendarAdapter(code)
    raise CalendarError(f"unknown calendar code {code!r}; known: {sorted(EXCHANGE_CODES | SYNTHETIC_CODES)}")


def get_calendar(code: str) -> TradingCalendar:
    """Return the (cached) calendar for an exchange code (XNYS, XLON, CMES, ...) or FX / CRYPTO."""
    return _build_calendar(str(code).strip().upper())


def union_sessions(codes: Iterable[str], start: DateLike, end: DateLike) -> pd.DatetimeIndex:
    """Sorted union of sessions across several calendars (a day counts if ANY market is open)."""
    indexes = [get_calendar(c).sessions(start, end) for c in codes]
    if not indexes:
        raise CalendarError("union_sessions needs at least one calendar code")
    return reduce(lambda a, b: a.union(b), indexes).sort_values()


def calendars_open_on(codes: Iterable[str], d: DateLike) -> list[str]:
    return [c for c in codes if get_calendar(c).is_session(d)]


def last_close(d: DateLike, codes: Iterable[str]) -> pd.Timestamp:
    """The latest close across the calendars that are open on `d` (the engine's decision time)."""
    codes = list(codes)
    open_codes = calendars_open_on(codes, d)
    if not open_codes:
        raise CalendarError(f"no market in {codes} is open on {to_date(d)}")
    return max(get_calendar(c).session_close(d) for c in open_codes)


def next_open(code: str, after: DateLike) -> tuple[date, pd.Timestamp]:
    """(session_date, open timestamp UTC) of the first session of `code` strictly after `after`."""
    cal = get_calendar(code)
    session = cal.next_session(after)
    return session, cal.session_open(session)
