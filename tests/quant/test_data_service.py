from __future__ import annotations

from datetime import date

import pytest
from conftest import FakeBarsAdapter, make_bars

from quant.config.settings import load_settings
from quant.core.errors import AdapterError, ConfigError
from quant.core.time import frozen_now
from quant.core.types import Instrument
from quant.data.service import DataService

S, E = date(2024, 6, 3), date(2024, 7, 19)  # 2024-07-19 is a Friday


@pytest.fixture
def fake(catalog):
    return FakeBarsAdapter(
        {
            "XNAS_AAPL": make_bars(catalog["XNAS_AAPL"], S, E, seed=1),
            "XLON_VOD": make_bars(catalog["XLON_VOD"], S, E, seed=2),
        }
    )


def _svc(tmp_path, fake, catalog, ttl_hours: float = 12.0) -> DataService:
    settings = load_settings(
        paths={"data_dir": str(tmp_path / "data")},
        data={"adapter_priority": ["fake"], "cache_ttl_hours": ttl_hours},
    )
    return DataService(settings, catalog=catalog, adapters={"fake": fake})


def test_fetch_writes_store_lineage_and_meta(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    report = svc.fetch_bars(["AAPL", "XLON_VOD", "XNYS_JPM"], S, E)
    assert report.vendor == "fake" and report.rows > 0
    assert sorted(w.instrument_id for w in report.writes) == ["XLON_VOD", "XNAS_AAPL"]
    assert report.errors == {"XNYS_JPM": "no fixture"}
    assert "2 error(s)" not in report.summary() and "1 error(s)" in report.summary()
    assert svc.parquet.exists("XNAS_AAPL", "fake")
    lineage = svc.lineage.recent()
    assert len(lineage) == 1 and lineage["operation"].iloc[0] == "fetch"
    assert lineage["lineage_id"].iloc[0] == report.lineage_id
    meta = svc.db.get_series_meta("XLON_VOD", "fake", "1d")
    assert meta["last_session"] == E and meta["rows"] == len(fake.series["XLON_VOD"])
    assert svc.db.get_series_meta("XNYS_JPM", "fake", "1d")["rows"] == 0
    assert set(svc.cached_series()["instrument_id"]) == {"XNAS_AAPL", "XLON_VOD", "XNYS_JPM"}


def test_get_bars_serves_from_cache_without_refetching(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    with frozen_now("2024-07-20T12:00:00Z"):
        first = svc.get_bars(["AAPL"], S, E)
        second = svc.get_bars(["AAPL"], S, E)
    assert len(first) == len(fake.series["XNAS_AAPL"]) and len(second) == len(first)
    assert len(fake.calls) == 1
    assert fake.calls[0] == (["XNAS_AAPL"], S, E)


def test_stale_cache_fetches_only_the_missing_tail(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog, ttl_hours=0)  # always stale
    with frozen_now("2024-07-20T12:00:00Z"):
        svc.fetch_bars(["AAPL"], S, date(2024, 7, 10))
        bars = svc.get_bars(["AAPL"], S, E)
    assert fake.calls[-1] == (["XNAS_AAPL"], date(2024, 7, 11), E)
    assert bars["session_date"].max().date() == E
    assert bars["session_date"].min().date() == S


def test_fresh_cache_skips_the_tail_within_ttl(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog, ttl_hours=12)
    with frozen_now("2024-07-20T12:00:00Z"):
        svc.fetch_bars(["AAPL"], S, date(2024, 7, 10))
        bars = svc.get_bars(["AAPL"], S, E)
    assert len(fake.calls) == 1  # checked moments ago: do not hammer the vendor
    assert bars["session_date"].max().date() == date(2024, 7, 10)


def test_missing_head_is_fetched(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    with frozen_now("2024-07-20T12:00:00Z"):
        svc.fetch_bars(["AAPL"], date(2024, 7, 1), E)
        bars = svc.get_bars(["AAPL"], S, E)
    assert fake.calls[-1] == (["XNAS_AAPL"], S, date(2024, 6, 30))
    assert bars["session_date"].min().date() == S


def test_refresh_false_never_touches_the_vendor(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    assert svc.get_bars(["AAPL"], S, E, refresh=False).empty
    assert fake.calls == []


def test_batches_instruments_sharing_a_range(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    with frozen_now("2024-07-20T12:00:00Z"):
        bars = svc.get_bars(["AAPL", "VOD.L"], S, E)
    assert len(fake.calls) == 1 and sorted(fake.calls[0][0]) == ["XLON_VOD", "XNAS_AAPL"]
    assert set(bars["instrument_id"]) == {"XNAS_AAPL", "XLON_VOD"}


def test_resolve_accepts_id_symbol_and_vendor_symbol(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    vod = catalog["XLON_VOD"]
    assert svc.resolve("XLON_VOD") == svc.resolve("VOD") == svc.resolve("VOD.L") == svc.resolve(vod) == vod
    assert svc.resolve("DGS10").instrument_id == "XNYS_US10Y"
    with pytest.raises(ConfigError, match="unknown instrument"):
        svc.resolve("NOPE")


def test_register_and_missing_capability(tmp_path, fake, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake, catalog)
    new = Instrument(symbol="TEST", asset_class="equity", market="XNYS", currency="USD")
    assert svc.register([new]) == 1
    assert svc.resolve("TEST") == new and svc.db.get_instrument("XNYS_TEST") == new
    with pytest.raises(AdapterError, match="no configured adapter provides macro"):
        svc.adapter_for("macro")
