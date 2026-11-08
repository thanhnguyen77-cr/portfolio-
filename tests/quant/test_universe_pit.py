from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant.core.errors import UniverseError
from quant.data.store import DuckDBStore
from quant.universe.pit import PITUniverse, _merge_adjacent


@pytest.fixture
def uni(tmp_path):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    return PITUniverse(db, "test")


def test_snapshots_open_and_close_intervals(uni):
    d1 = uni.add_snapshot(date(2024, 1, 2), ["B", "A"], "s1")
    assert d1.added == ("A", "B") and d1.removed == () and d1.members == 2
    assert uni.quality() == "current_only" and uni.history_start() == date(2024, 1, 2)
    d2 = uni.add_snapshot(date(2024, 2, 1), ["A", "C"], "s2")
    assert d2.added == ("C",) and d2.removed == ("B",)
    iv = uni.intervals().set_index("instrument_id")
    assert iv.loc["A", "effective_from"] == date(2024, 1, 2) and iv.loc["A", "effective_to"] is None
    assert iv.loc["B", "effective_to"] == date(2024, 1, 31)
    assert iv.loc["C", "effective_from"] == date(2024, 2, 1)
    assert uni.membership_at(date(2024, 1, 15)) == ["A", "B"]
    assert uni.membership_at(date(2024, 2, 1)) == ["A", "C"]
    assert uni.membership_at(date(2023, 12, 31)) == []
    assert uni.snapshots()["members"].tolist() == [2, 2]
    with pytest.raises(UniverseError, match="older than"):
        uni.add_snapshot(date(2024, 1, 20), ["A"], "late")
    with pytest.raises(UniverseError, match="at least one"):
        uni.add_snapshot(date(2024, 3, 1), [], "empty")


def test_members_over_and_coverage(uni):
    uni.add_snapshot(date(2024, 1, 2), ["A", "B"], "s1")
    uni.add_snapshot(date(2024, 2, 1), ["A", "C"], "s2")
    assert uni.members_over(date(2024, 1, 1), date(2024, 3, 1)) == ["A", "B", "C"]
    assert uni.members_over(date(2024, 2, 15), date(2024, 3, 1)) == ["A", "C"]
    assert uni.coverage(date(2024, 1, 1), date(2024, 3, 1), with_data={"A", "C"}) == pytest.approx(2 / 3)
    assert PITUniverse(uni.db, "empty").coverage(date(2024, 1, 1), date(2024, 3, 1), {"A"}) == 0.0


@settings(max_examples=25, deadline=None)
@given(
    st.lists(st.sets(st.sampled_from(["A", "B", "C", "D", "E"]), min_size=1), min_size=1, max_size=6),
    st.lists(st.integers(min_value=1, max_value=40), min_size=6, max_size=6),
)
def test_snapshot_sequences_never_overlap_and_reproduce_each_snapshot(snapshots, gaps):
    with tempfile.TemporaryDirectory() as tmp:
        db = DuckDBStore(Path(tmp) / "q.duckdb")
        db.migrate()
        uni = PITUniverse(db, "prop")
        day = date(2024, 1, 2)
        days = []
        for members, gap in zip(snapshots, gaps, strict=False):
            uni.add_snapshot(day, members, "s")
            days.append((day, set(members)))
            day += timedelta(days=gap)
        for d, members in days:
            assert set(uni.membership_at(d)) == members
        iv = uni.intervals()
        for _, g in iv.groupby("instrument_id"):
            g = g.sort_values("effective_from")
            ends = [e if e is not None else date.max for e in g["effective_to"]]
            for i in range(1, len(g)):
                assert g["effective_from"].iloc[i] > ends[i - 1]
            assert (iv["effective_to"].isna() | (iv["effective_to"] >= iv["effective_from"])).all()


def test_reconstruct_from_changes_hand_built(uni):
    changes = pd.DataFrame(
        {
            "date": [date(2024, 6, 20), date(2024, 3, 18)],
            "added_id": ["C", "B"],
            "removed_id": ["D", "E"],
        }
    )
    n = uni.reconstruct_from_changes(["A", "B", "C"], date(2024, 7, 1), changes)
    assert n == 5 and uni.quality() == "reconstructed" and uni.history_start() == date(2024, 3, 17)
    iv = uni.intervals().set_index("instrument_id")
    assert iv.loc["A", "effective_from"] == date(2024, 3, 17) and iv.loc["A", "effective_to"] is None
    assert iv.loc["B", "effective_from"] == date(2024, 3, 18)
    assert iv.loc["C", "effective_from"] == date(2024, 6, 20)
    assert (iv.loc["D", "effective_from"], iv.loc["D", "effective_to"]) == (
        date(2024, 3, 17),
        date(2024, 6, 19),
    )
    assert (iv.loc["E", "effective_from"], iv.loc["E", "effective_to"]) == (
        date(2024, 3, 17),
        date(2024, 3, 17),
    )
    assert uni.membership_at(date(2024, 3, 17)) == ["A", "D", "E"]
    assert uni.membership_at(date(2024, 3, 18)) == ["A", "B", "D"]
    assert uni.membership_at(date(2024, 6, 20)) == ["A", "B", "C"]
    assert uni.membership_at(date(2024, 3, 16)) == []


def test_reconstruct_merges_a_rename_into_one_interval(uni):
    changes = pd.DataFrame({"date": [date(2024, 6, 20)], "added_id": ["META"], "removed_id": ["META"]})
    uni.reconstruct_from_changes(["META", "A"], date(2024, 7, 1), changes)
    iv = uni.intervals()
    assert (iv["instrument_id"] == "META").sum() == 1
    meta = iv[iv["instrument_id"] == "META"].iloc[0]
    assert meta["effective_from"] == date(2024, 6, 19) and meta["effective_to"] is None


def test_readd_after_removal_keeps_two_stints(uni):
    changes = pd.DataFrame(
        {"date": [date(2024, 6, 1), date(2024, 2, 1)], "added_id": ["X", None], "removed_id": [None, "X"]}
    )
    uni.reconstruct_from_changes(["X"], date(2024, 7, 1), changes)
    iv = uni.intervals()
    assert iv["effective_from"].tolist() == [date(2024, 1, 31), date(2024, 6, 1)]
    assert iv["effective_to"].tolist() == [date(2024, 1, 31), None]


def test_import_intervals_and_quality(uni):
    df = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "B"],
            "effective_from": ["2020-01-01", "2021-01-01", "2022-06-01"],
            "effective_to": ["2020-12-31", None, None],
        }
    )
    assert uni.import_intervals(df, "csv", complete=True) == 2  # A's adjacent stints merged
    assert uni.quality() == "pit_complete" and uni.history_start() == date(2020, 1, 1)
    assert uni.membership_at(date(2020, 6, 1)) == ["A"] and uni.membership_at(date(2023, 1, 1)) == ["A", "B"]
    with pytest.raises(UniverseError, match="needs columns"):
        uni.import_intervals(pd.DataFrame({"x": [1]}), "csv")
    with pytest.raises(UniverseError, match="quality must be"):
        uni.set_quality("perfect")


def test_merge_adjacent():
    rows = [
        ("A", date(2024, 1, 1), date(2024, 1, 31)),
        ("A", date(2024, 2, 1), None),
        ("B", date(2024, 1, 1), date(2024, 1, 5)),
        ("B", date(2024, 1, 10), date(2024, 1, 20)),
    ]
    merged = _merge_adjacent(rows)
    assert merged == [
        ("A", date(2024, 1, 1), None),
        ("B", date(2024, 1, 1), date(2024, 1, 5)),
        ("B", date(2024, 1, 10), date(2024, 1, 20)),
    ]
