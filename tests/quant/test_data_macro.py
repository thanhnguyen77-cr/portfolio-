from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES, FakeMacroAdapter

from quant.config.settings import load_settings
from quant.core.time import frozen_now
from quant.data.adapters.boe import BankOfEnglandAdapter
from quant.data.adapters.fred import FREDAdapter
from quant.data.service import DataService

S, E = date(2024, 7, 1), date(2024, 7, 5)


class _Resp:
    def __init__(self, text: str = "", payload=None, status: int = 200):
        self.text, self._payload, self.status_code = text, payload, status

    def json(self):
        return self._payload


def _stub_http(adapter, response=None, exc: Exception | None = None, calls: list | None = None):
    def _http(method, url, **kw):
        if calls is not None:
            calls.append({"method": method, "url": url, **kw})
        if exc is not None:
            raise exc
        return response

    adapter._http = _http  # the network seam
    return adapter


def test_fred_csv_parse_drops_missing_markers():
    adapter = _stub_http(FREDAdapter(), _Resp(text=(FIXTURES / "fred_DGS10.csv").read_text()))
    res = adapter.get_macro_series(["DGS10"], S, E)
    assert res.errors == {} and len(res.frame) == 4  # the "." on July 4th is dropped
    assert res.frame["value"].tolist() == [4.48, 4.43, 4.36, 4.28]
    assert (res.frame["vendor"] == "fred").all() and res.frame["date"].iloc[0] == pd.Timestamp("2024-07-01")


def test_fred_legacy_header_and_range_filter():
    text = "DATE,DGS10\n2024-07-01,4.48\n2024-07-02,4.43\n2024-07-03,4.36\n"
    adapter = _stub_http(FREDAdapter(), _Resp(text=text))
    res = adapter.get_macro_series(["DGS10"], date(2024, 7, 2), date(2024, 7, 3))
    assert res.frame["value"].tolist() == [4.43, 4.36]


def test_fred_api_path_when_key_is_configured():
    calls: list = []
    observations = [{"date": "2024-07-01", "value": "4.48"}, {"date": "2024-07-02", "value": "."}]
    payload = {"observations": observations}
    adapter = _stub_http(FREDAdapter(api_key="secret"), _Resp(payload=payload), calls=calls)
    res = adapter.get_macro_series(["DGS10"], S, E)
    assert res.frame["value"].tolist() == [4.48] and res.params["api"] is True
    params = calls[0]["params"]
    assert params["api_key"] == "secret" and params["observation_start"] == "2024-07-01"
    assert "api.stlouisfed.org" in calls[0]["url"]


def test_fred_per_series_errors():
    adapter = _stub_http(FREDAdapter(), exc=RuntimeError("boom"))
    res = adapter.get_macro_series(["DGS10", "DGS2"], S, E)
    assert res.frame.empty and set(res.errors) == {"DGS10", "DGS2"} and "RuntimeError" in res.errors["DGS2"]
    assert res.all_failed


def test_boe_multi_series_parse_and_params():
    calls: list = []
    text = (FIXTURES / "boe_IUDBEDR_IUDSOIA.csv").read_text()
    adapter = _stub_http(BankOfEnglandAdapter(), _Resp(text=text), calls=calls)
    res = adapter.get_macro_series(["IUDBEDR", "iudsoia", "NOPE"], S, E)
    assert len(res.frame) == 10 and set(res.frame["series_id"]) == {"IUDBEDR", "IUDSOIA"}
    assert res.frame[res.frame["series_id"] == "IUDSOIA"]["value"].tolist()[2] == 5.19
    assert res.frame["date"].min() == pd.Timestamp("2024-07-01")
    assert "NOPE" in res.errors and "not in response" in res.errors["NOPE"]
    params = calls[0]["params"]
    assert params["Datefrom"] == "01/Jul/2024" and params["Dateto"] == "05/Jul/2024"
    assert params["SeriesCodes"] == "IUDBEDR,IUDSOIA,NOPE" and params["csv.x"] == "yes"


def test_boe_batch_failure_reports_every_series():
    adapter = _stub_http(BankOfEnglandAdapter(), exc=RuntimeError("down"))
    res = adapter.get_macro_series(["IUDBEDR", "IUDSOIA"], S, E)
    assert res.frame.empty and set(res.errors) == {"IUDBEDR", "IUDSOIA"}


@pytest.fixture
def fake_macro():
    idx = pd.bdate_range("2024-06-03", "2024-07-19")
    return FakeMacroAdapter({"DGS10": pd.Series(4.0 + 0.01 * np.arange(len(idx)), index=idx)})


def _svc(tmp_path, fake_macro, catalog, ttl_hours: float = 12.0) -> DataService:
    settings = load_settings(
        paths={"data_dir": str(tmp_path / "data")},
        data={"adapter_priority": ["fakemacro"], "cache_ttl_hours": ttl_hours},
    )
    return DataService(settings, catalog=catalog, adapters={"fakemacro": fake_macro})


def test_service_get_macro_caches_and_records(tmp_path, fake_macro, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake_macro, catalog)
    with frozen_now("2024-07-20T12:00:00Z"):
        s1 = svc.get_macro("DGS10", date(2024, 6, 3), date(2024, 7, 19))
        s2 = svc.get_macro("DGS10", date(2024, 6, 3), date(2024, 7, 19))
    assert len(s1) == len(pd.bdate_range("2024-06-03", "2024-07-19")) and s1.name == "DGS10"
    assert s2.equals(s1) and len(fake_macro.calls) == 1
    lineage = svc.lineage.recent()
    assert lineage["operation"].tolist() == ["fetch_macro"]
    meta = svc.db.get_series_meta("DGS10", "fakemacro", "macro")
    assert meta["last_session"] == date(2024, 7, 19) and meta["rows"] == len(s1)


def test_service_get_macro_refreshes_tail_when_stale(tmp_path, fake_macro, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    svc = _svc(tmp_path, fake_macro, catalog, ttl_hours=0)
    with frozen_now("2024-07-20T12:00:00Z"):
        svc.fetch_macro(["DGS10"], date(2024, 6, 3), date(2024, 7, 10))
        s = svc.get_macro("DGS10", date(2024, 6, 3), date(2024, 7, 19))
    assert fake_macro.calls[-1] == (["DGS10"], date(2024, 7, 11), date(2024, 7, 19))
    assert s.index.max() == pd.Timestamp("2024-07-19")
    assert svc.get_macro("NOPE", date(2024, 6, 3), date(2024, 7, 19)).empty
