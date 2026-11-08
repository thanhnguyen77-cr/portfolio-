from __future__ import annotations

from datetime import date

import pandas as pd
from conftest import make_bars

from quant.data.lineage import LineageLog, LineageRecord, frame_digest, package_versions
from quant.data.store import DuckDBStore


def test_frame_digest_is_deterministic_and_sensitive(catalog):
    a = make_bars(catalog["XNAS_AAPL"], date(2024, 6, 3), date(2024, 6, 28))
    assert frame_digest(a) == frame_digest(a.copy())
    b = a.copy()
    b.loc[0, "close"] += 0.01
    assert frame_digest(a) != frame_digest(b)
    assert frame_digest(a.iloc[0:0]) == frame_digest(pd.DataFrame())


def test_package_versions_include_core_libraries():
    v = package_versions()
    assert {"quant", "pandas", "numpy", "pyarrow", "duckdb"} <= set(v)


def test_record_and_query(tmp_path):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    log = LineageLog(db)
    rec = LineageRecord(
        vendor="fake",
        operation="fetch",
        instrument_ids=["XNAS_AAPL", "XLON_VOD"],
        rows=10,
        params={"start": "2024-01-01"},
        diff={"writes": 2},
    )
    log.record(rec)
    log.record(LineageRecord(vendor="fake", operation="fetch", instrument_ids=["FX_GBPUSD"], rows=3))
    recent = log.recent()
    assert len(recent) == 2 and set(recent["operation"]) == {"fetch"}
    mine = log.for_instrument("XLON_VOD")
    assert len(mine) == 1 and mine["lineage_id"].iloc[0] == rec.lineage_id
    assert '"start": "2024-01-01"' in mine["params"].iloc[0]
    assert log.for_instrument("NOPE").empty
