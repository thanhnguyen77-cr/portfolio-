from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from quant.core.time import ensure_utc, frozen_now, now, to_date, today


def test_now_is_tz_aware_utc():
    t = now()
    assert t.tzinfo is not None
    assert t.utcoffset().total_seconds() == 0


def test_frozen_now_and_restore():
    with frozen_now("2024-01-02T10:00:00Z") as fixed:
        assert fixed == datetime(2024, 1, 2, 10, tzinfo=UTC)
        assert now() == fixed
        assert today() == date(2024, 1, 2)
    assert now() != fixed


def test_frozen_now_nests():
    with frozen_now("2024-01-01"):
        with frozen_now("2025-06-01"):
            assert today() == date(2025, 6, 1)
        assert today() == date(2024, 1, 1)


def test_ensure_utc_treats_naive_as_utc_and_converts_aware():
    assert ensure_utc(datetime(2024, 1, 1, 12)) == datetime(2024, 1, 1, 12, tzinfo=UTC)
    ny = pd.Timestamp("2024-01-01 12:00", tz="America/New_York")
    assert ensure_utc(ny) == pd.Timestamp("2024-01-01 17:00", tz="UTC")
    assert ensure_utc(pd.Timestamp("2024-01-01 12:00")) == pd.Timestamp("2024-01-01 12:00", tz="UTC")


def test_ensure_utc_rejects_strings():
    with pytest.raises(TypeError):
        ensure_utc("2024-01-01")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        "2024-03-05",
        date(2024, 3, 5),
        datetime(2024, 3, 5, 23, 59),
        pd.Timestamp("2024-03-05 01:00", tz="Asia/Tokyo"),
    ],
)
def test_to_date_variants(value):
    assert to_date(value) == date(2024, 3, 5)


def test_to_date_rejects_int():
    with pytest.raises(TypeError):
        to_date(20240305)  # type: ignore[arg-type]
