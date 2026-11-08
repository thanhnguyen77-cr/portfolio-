from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import actions_frame, make_bars, plant

from quant.data.validation import ValidationConfig, validate_bars

S, E = date(2024, 1, 2), date(2024, 12, 31)


@pytest.fixture
def aapl(catalog):
    return make_bars(catalog["XNAS_AAPL"], S, E, seed=4)


def _report(bars, catalog, actions=None, **cfg):
    return validate_bars(bars, catalog, actions, ValidationConfig(**cfg) if cfg else None)


def test_clean_series_grades_a(aapl, catalog):
    r = _report(aapl, catalog)
    q = r["XNAS_AAPL"]
    assert q.grade == "A" and q.score == 100.0
    assert (q.gaps, q.extra_bars, q.stale_days, q.outliers, q.ohlc_violations) == (0, 0, 0, 0, 0)
    assert q.denomination_changes == 0 and q.unexplained_jumps == 0 and q.issues == []
    assert q.expected_sessions == q.bars == len(aapl)
    assert r.ok("A") and r.worst_grade() == "A"


def test_gaps_are_counted_and_listed(aapl, catalog):
    q = _report(plant(aapl, drop_dates=[date(2024, 3, 5), date(2024, 3, 6)]), catalog)["XNAS_AAPL"]
    assert q.gaps == 2 and q.gap_pct == pytest.approx(2 / q.expected_sessions)
    assert sorted(i.date for i in q.issues if i.kind == "gap") == [date(2024, 3, 5), date(2024, 3, 6)]
    assert q.score < 100


def test_stale_runs(aapl, catalog):
    q = _report(plant(aapl, stale=(50, 5)), catalog)["XNAS_AAPL"]
    assert q.max_stale_run == 5 and q.stale_days == 4
    assert any(i.kind == "stale" for i in q.issues)
    short_run = _report(plant(aapl, stale=(50, 2)), catalog)["XNAS_AAPL"]
    assert short_run.stale_days == 0  # below the run threshold


def test_outlier_spike(aapl, catalog):
    planted = plant(aapl, outlier=(100, 1.5))
    q = _report(planted, catalog)["XNAS_AAPL"]
    assert q.outliers >= 1
    assert planted.loc[100, "session_date"].date() in {i.date for i in q.issues if i.kind == "outlier"}


def test_denomination_change_is_not_a_split(aapl, catalog):
    q = _report(plant(aapl, denomination=120), catalog)["XNAS_AAPL"]
    assert q.denomination_changes == 1 and q.unexplained_jumps == 0 and q.explained_splits == 0
    assert q.grade != "A" and any(i.kind == "denomination_change" for i in q.issues)


def test_split_jump_needs_a_recorded_action(aapl, catalog):
    planted = plant(aapl, jump=(120, 0.5))
    ex = planted.loc[120, "session_date"]
    without = _report(planted, catalog)["XNAS_AAPL"]
    assert without.unexplained_jumps == 1 and without.explained_splits == 0
    acts = actions_frame("XNAS_AAPL", [(ex, "split", 2.0, None)])
    with_split = _report(planted, catalog, acts)["XNAS_AAPL"]
    assert with_split.unexplained_jumps == 0 and with_split.explained_splits == 1
    assert with_split.score > without.score


def test_ohlc_violation(aapl, catalog):
    q = _report(plant(aapl, ohlc=30), catalog)["XNAS_AAPL"]
    assert q.ohlc_violations == 1 and any(i.kind == "ohlc" and i.severity == "error" for i in q.issues)


def test_dividend_sanity(aapl, catalog):
    ex = pd.Timestamp("2024-06-17")
    ok = actions_frame("XNAS_AAPL", [(ex, "dividend", 1.0, "USD")])
    assert _report(aapl, catalog, ok)["XNAS_AAPL"].dividend_warnings == 0
    huge = actions_frame("XNAS_AAPL", [(ex, "dividend", 60.0, "USD")])  # ~60% of a ~100 price: unit error
    assert _report(aapl, catalog, huge)["XNAS_AAPL"].dividend_warnings == 1
    neg = actions_frame("XNAS_AAPL", [(ex, "dividend", -1.0, "USD")])
    assert _report(aapl, catalog, neg)["XNAS_AAPL"].dividend_warnings == 1


def test_extra_bar_on_a_holiday(aapl, catalog):
    holiday = aapl.iloc[[100]].copy()
    holiday["session_date"] = pd.Timestamp("2024-07-04")
    q = _report(pd.concat([aapl, holiday]), catalog)["XNAS_AAPL"]
    assert q.extra_bars == 1 and any(i.kind == "extra_bar" for i in q.issues)


def test_zero_volume_only_matters_for_traded_assets(aapl, catalog):
    silent = aapl.copy()
    silent["volume"] = 0.0
    q = _report(silent, catalog)["XNAS_AAPL"]
    assert q.zero_volume_pct == 1.0 and any(i.kind == "volume" for i in q.issues)
    fx = make_bars(catalog["FX_GBPUSD"], S, E, seed=2)
    fx["volume"] = 0.0
    assert _report(fx, catalog)["FX_GBPUSD"].zero_volume_pct == 0.0


def test_report_frames_and_ordering(aapl, catalog):
    vod = plant(make_bars(catalog["XLON_VOD"], S, E, seed=5), denomination=50, stale=(10, 6))
    r = _report(pd.concat([aapl, vod]), catalog)
    table = r.to_frame()
    assert table["instrument_id"].tolist() == ["XLON_VOD", "XNAS_AAPL"]  # worst first
    assert r.worst_grade() in {"B", "C", "F"} and not r.ok("A")
    issues = r.issues_frame()
    assert set(issues["instrument_id"]) == {"XLON_VOD"}
    assert {"kind", "severity", "date", "detail"} <= set(issues.columns)


def test_empty_input(catalog):
    r = validate_bars(pd.DataFrame(), catalog)
    assert r.to_frame().empty and r.ok() and r.worst_grade() == "A"
