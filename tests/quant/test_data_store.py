from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import make_bars

from quant.core.errors import StoreError
from quant.core.types import Instrument
from quant.data.store import DuckDBStore, ParquetStore, coerce_bars, dedupe_bars

START, END = date(2024, 6, 3), date(2024, 7, 12)


@pytest.fixture
def aapl(catalog) -> Instrument:
    return catalog["XNAS_AAPL"]


@pytest.fixture
def vod(catalog) -> Instrument:
    return catalog["XLON_VOD"]


def test_coerce_bars_normalises_dtypes_and_sorts(aapl):
    bars = make_bars(aapl, START, END).sample(frac=1, random_state=1)  # shuffled
    out = coerce_bars(bars)
    assert str(out["session_date"].dtype) == "datetime64[ns]"
    assert str(out["ts_close_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(out["ingested_at"].dtype) == "datetime64[ns, UTC]"
    assert out["close"].dtype == "float64"
    assert out["session_date"].is_monotonic_increasing
    assert out["instrument_id"].eq("XNAS_AAPL").all()


def test_coerce_bars_rejects_bad_frames(aapl):
    bars = make_bars(aapl, START, END)
    with pytest.raises(StoreError, match="missing columns"):
        coerce_bars(bars.drop(columns=["volume"]))
    bad = bars.copy()
    bad.loc[0, "close"] = float("nan")
    with pytest.raises(StoreError, match="without a close"):
        coerce_bars(bad)
    bad = bars.copy()
    bad.loc[0, "currency"] = None
    with pytest.raises(StoreError, match="nulls"):
        coerce_bars(bad)
    bad = bars.copy()
    bad["session_date"] = bad["session_date"].dt.tz_localize("UTC")
    with pytest.raises(StoreError, match="tz-naive"):
        coerce_bars(bad)
    assert coerce_bars(bars.iloc[0:0]).empty


def test_parquet_round_trip_preserves_values_and_dtypes(tmp_path, aapl):
    store = ParquetStore(tmp_path)
    bars = make_bars(aapl, START, END)
    [res] = store.write_bars(bars)
    assert res.rows_before == 0 and res.rows_after == len(bars) and res.rows_new == len(bars)
    assert res.path == tmp_path / "vendor=synthetic" / "freq=1d" / "XNAS_AAPL.parquet"
    back = store.read_series("XNAS_AAPL", "synthetic")
    pd.testing.assert_frame_equal(back, bars)
    assert store.exists("XNAS_AAPL", "synthetic")
    assert store.list_instruments("synthetic") == ["XNAS_AAPL"]
    assert store.last_session("XNAS_AAPL", "synthetic") == date(2024, 7, 12)
    assert store.last_session("XNAS_AAPL", "other") is None


def test_append_is_idempotent(tmp_path, aapl):
    store = ParquetStore(tmp_path)
    bars = make_bars(aapl, START, END)
    store.write_bars(bars)
    [again] = store.write_bars(bars)
    assert (again.rows_new, again.rows_updated, again.rows_unchanged) == (0, 0, len(bars))
    assert again.rows_after == len(bars)
    pd.testing.assert_frame_equal(store.read_series("XNAS_AAPL", "synthetic"), bars)


def test_restatement_latest_ingested_at_wins(tmp_path, aapl):
    store = ParquetStore(tmp_path)
    bars = make_bars(aapl, START, END)
    store.write_bars(bars)

    newer = bars.iloc[[5]].copy()
    newer["close"] = 999.0
    newer["ingested_at"] = bars["ingested_at"].iloc[0] + pd.Timedelta(hours=1)
    [res] = store.write_bars(newer)
    assert (res.rows_new, res.rows_updated, res.rows_unchanged) == (0, 1, 0)
    assert store.read_series("XNAS_AAPL", "synthetic").loc[5, "close"] == 999.0

    older = bars.iloc[[6]].copy()
    older["close"] = -1.0
    older["ingested_at"] = bars["ingested_at"].iloc[0] - pd.Timedelta(days=1)
    store.write_bars(older)
    assert store.read_series("XNAS_AAPL", "synthetic").loc[6, "close"] == bars.loc[6, "close"]
    assert len(store.read_series("XNAS_AAPL", "synthetic")) == len(bars)


def test_dedupe_bars_prefers_latest_ingested():
    a = pd.DataFrame(
        {
            "instrument_id": ["X", "X"],
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "close": [1.0, 2.0],
            "ingested_at": pd.to_datetime(["2024-01-03", "2024-01-02"], utc=True),
        }
    )
    out = dedupe_bars(a.iloc[:0], a)
    assert len(out) == 1 and out["close"].iloc[0] == 1.0


def test_read_bars_filters_and_combines(tmp_path, aapl, vod):
    store = ParquetStore(tmp_path)
    store.write_bars(pd.concat([make_bars(aapl, START, END), make_bars(vod, START, END, seed=3)]))
    out = store.read_bars(
        ["XNAS_AAPL", "XLON_VOD", "XNAS_MISSING"], "synthetic", date(2024, 7, 1), date(2024, 7, 5)
    )
    assert set(out["instrument_id"]) == {"XNAS_AAPL", "XLON_VOD"}
    aapl_days = out[out["instrument_id"] == "XNAS_AAPL"]["session_date"].dt.date.tolist()
    assert aapl_days == [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3), date(2024, 7, 5)]
    vod_days = out[out["instrument_id"] == "XLON_VOD"]["session_date"].dt.date.tolist()
    assert date(2024, 7, 4) in vod_days  # London open on US Independence Day
    assert store.read_bars(["XNAS_MISSING"], "synthetic").empty
    assert store.delete_series("XLON_VOD", "synthetic") and not store.delete_series("XLON_VOD", "synthetic")


def test_unsafe_instrument_id_rejected(tmp_path):
    with pytest.raises(StoreError, match="filesystem-safe"):
        ParquetStore(tmp_path).path_for("../evil", "v")


def test_duckdb_migrate_is_idempotent(tmp_path):
    from quant.data.store import SCHEMA_VERSION

    db = DuckDBStore(tmp_path / "q.duckdb")
    assert db.migrate() == SCHEMA_VERSION
    assert db.migrate() == SCHEMA_VERSION
    assert {"instruments", "lineage", "series_meta", "symbol_history", "schema_version"} <= set(db.tables())
    assert {"paper_accounts", "paper_orders", "paper_fills", "paper_events"} <= set(db.tables())
    assert db.query("SELECT count(*) AS n FROM schema_version")["n"].iloc[0] == 1


def test_duckdb_read_before_migrate_raises(tmp_path):
    with pytest.raises(StoreError, match="migrate"):
        DuckDBStore(tmp_path / "missing.duckdb").query("SELECT 1")


def test_instruments_round_trip(tmp_path, catalog):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    subset = [catalog["XNAS_AAPL"], catalog["XLON_VOD"], catalog["FX_GBPUSD"]]
    assert db.upsert_instruments(subset) == 3
    assert db.upsert_instruments(subset) == 3  # replace, no duplicates
    assert db.get_instrument("XLON_VOD") == catalog["XLON_VOD"]
    assert db.get_instrument("NOPE") is None
    assert [i.instrument_id for i in db.list_instruments()] == ["FX_GBPUSD", "XLON_VOD", "XNAS_AAPL"]


def test_series_meta_tracks_checks_and_writes(tmp_path):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    span = {"first_session": date(2024, 1, 2), "last_session": date(2024, 1, 31), "rows": 21}
    db.upsert_series_meta("X", "v", "1d", written=True, **span)
    first = db.get_series_meta("X", "v", "1d")
    assert first["rows"] == 21 and first["last_written_at"] is not None
    db.upsert_series_meta("X", "v", "1d", written=False, **span)
    second = db.get_series_meta("X", "v", "1d")
    assert pd.Timestamp(second["last_written_at"]) == pd.Timestamp(first["last_written_at"])
    assert pd.Timestamp(second["last_checked_at"]) >= pd.Timestamp(first["last_checked_at"])
    assert db.get_series_meta("Y", "v", "1d") is None


def test_bars_view_over_parquet(tmp_path, aapl, vod):
    store = ParquetStore(tmp_path / "pq")
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    assert db.register_bars_view(store.root) is False
    a, v = make_bars(aapl, START, END), make_bars(vod, START, END, seed=9)
    store.write_bars(pd.concat([a, v]))
    assert db.register_bars_view(store.root) is True
    counts = db.query("SELECT instrument_id, vendor, count(*) AS n FROM bars GROUP BY 1, 2 ORDER BY 1")
    assert counts["n"].tolist() == [len(v), len(a)]
    assert counts["vendor"].eq("synthetic").all()
