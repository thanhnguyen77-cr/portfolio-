from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant.core.calendars import (
    calendars_open_on,
    get_calendar,
    last_close,
    next_open,
    union_sessions,
)
from quant.core.errors import CalendarError


def _utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def test_get_calendar_is_cached_and_validates_code():
    assert get_calendar("XNYS") is get_calendar("xnys")
    with pytest.raises(CalendarError, match="unknown calendar code"):
        get_calendar("NOPE")


def test_xnys_holidays_and_sessions():
    cal = get_calendar("XNYS")
    assert not cal.is_session(date(2024, 7, 4))  # Independence Day
    assert cal.is_session(date(2024, 7, 5))
    assert cal.is_session(date(2024, 12, 26))  # not a US holiday
    assert not cal.is_session(date(2024, 7, 6))  # Saturday


def test_xlon_holidays_and_sessions():
    cal = get_calendar("XLON")
    assert not cal.is_session(date(2024, 8, 26))  # Summer bank holiday
    assert not cal.is_session(date(2024, 12, 26))  # Boxing Day
    assert cal.is_session(date(2024, 7, 4))  # London is open on US Independence Day


def test_xlon_close_follows_british_summer_time():
    cal = get_calendar("XLON")
    assert cal.session_close(date(2024, 1, 15)) == _utc("2024-01-15 16:30")  # GMT
    assert cal.session_close(date(2024, 7, 15)) == _utc("2024-07-15 15:30")  # BST
    assert cal.session_open(date(2024, 7, 15)) == _utc("2024-07-15 07:00")


def test_xnys_open_close_dst_and_early_close():
    cal = get_calendar("XNYS")
    assert cal.session_close(date(2024, 1, 16)) == _utc("2024-01-16 21:00")  # EST
    assert cal.session_close(date(2024, 7, 16)) == _utc("2024-07-16 20:00")  # EDT
    assert cal.session_open(date(2024, 7, 16)) == _utc("2024-07-16 13:30")
    assert cal.session_close(date(2024, 7, 3)) == _utc("2024-07-03 17:00")  # 1pm early close


def test_sessions_are_naive_dates_and_next_previous():
    cal = get_calendar("XNYS")
    s = cal.sessions(date(2024, 7, 1), date(2024, 7, 5))
    assert s.tz is None
    assert list(s.date) == [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5)]
    assert cal.next_session(date(2024, 7, 3)) == date(2024, 7, 5)
    assert cal.next_session(date(2024, 7, 4)) == date(2024, 7, 5)
    assert cal.previous_session(date(2024, 7, 5)) == date(2024, 7, 3)
    assert cal.date_to_session(date(2024, 7, 4), "previous") == date(2024, 7, 3)
    assert cal.date_to_session(date(2024, 7, 4), "next") == date(2024, 7, 5)


def test_non_session_open_close_raises():
    with pytest.raises(CalendarError, match="not a trading session"):
        get_calendar("XNYS").session_close(date(2024, 7, 4))


def test_out_of_bounds_raises():
    with pytest.raises(CalendarError, match="outside calendar bounds"):
        get_calendar("XNYS").is_session(date(1900, 1, 1))


def test_union_sessions_and_last_close():
    codes = ["XNYS", "XLON"]
    u = union_sessions(codes, date(2024, 7, 1), date(2024, 7, 5))
    assert pd.Timestamp("2024-07-04") in u  # London open while New York is shut
    assert len(u) == 5
    assert calendars_open_on(codes, date(2024, 7, 4)) == ["XLON"]
    assert last_close(date(2024, 7, 4), codes) == _utc("2024-07-04 15:30")
    assert last_close(date(2024, 7, 2), codes) == _utc("2024-07-02 20:00")
    with pytest.raises(CalendarError, match="no market"):
        last_close(date(2024, 7, 6), codes)


def test_next_open():
    assert next_open("XNYS", date(2024, 7, 3)) == (date(2024, 7, 5), _utc("2024-07-05 13:30"))
    assert next_open("XLON", date(2024, 8, 23)) == (date(2024, 8, 27), _utc("2024-08-27 07:00"))


def test_fx_calendar_weekdays_only_with_new_york_close():
    cal = get_calendar("FX")
    assert len(cal.sessions(date(2024, 7, 1), date(2024, 7, 7))) == 5
    assert not cal.is_session(date(2024, 7, 6))
    assert cal.is_session(date(2024, 7, 4))  # FX trades through US holidays
    assert cal.session_close(date(2024, 7, 15)) == _utc("2024-07-15 21:00")  # 17:00 EDT
    assert cal.session_close(date(2024, 1, 15)) == _utc("2024-01-15 22:00")  # 17:00 EST
    assert cal.session_open(date(2024, 7, 15)) == _utc("2024-07-14 21:00")
    assert cal.next_session(date(2024, 7, 5)) == date(2024, 7, 8)
    with pytest.raises(CalendarError):
        cal.session_close(date(2024, 7, 6))


def test_crypto_calendar_every_day():
    cal = get_calendar("CRYPTO")
    assert len(cal.sessions(date(2024, 7, 1), date(2024, 7, 7))) == 7
    assert cal.is_session(date(2024, 7, 6))
    assert cal.session_open(date(2024, 7, 6)) == _utc("2024-07-06 00:00")
    assert cal.session_close(date(2024, 7, 6)) == _utc("2024-07-06 23:59:59")
    assert cal.next_session(date(2024, 7, 6)) == date(2024, 7, 7)


@pytest.mark.parametrize("code", ["XNAS", "CMES", "XCBT", "XCEC", "XNYM"])
def test_futures_and_nasdaq_calendars_build(code):
    cal = get_calendar(code)
    assert cal.is_session(date(2024, 7, 5))
