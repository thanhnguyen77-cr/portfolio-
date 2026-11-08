from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import requests
from conftest import FakeBarsAdapter

from quant.core.errors import AdapterError
from quant.data.adapters.base import (
    AdapterResult,
    BaseAdapter,
    BaseHTTPAdapter,
    Capability,
    MarketDataAdapter,
)
from quant.data.adapters.yfinance_adapter import YFinanceAdapter

D = date(2024, 1, 2)


def test_adapters_satisfy_the_protocol():
    assert isinstance(YFinanceAdapter(history_fn=lambda *a: pd.DataFrame()), MarketDataAdapter)
    assert isinstance(FakeBarsAdapter({}), MarketDataAdapter)
    assert isinstance(BaseAdapter(), MarketDataAdapter)


def test_base_adapter_declares_nothing_and_rejects_everything():
    base = BaseAdapter()
    assert base.capabilities() == frozenset() and not base.supports("bars_daily")
    with pytest.raises(AdapterError, match="does not provide bars_daily"):
        base.get_bars([], D, D)
    with pytest.raises(AdapterError, match="corporate_actions"):
        base.get_corporate_actions([], D, D)
    with pytest.raises(AdapterError, match="macro"):
        base.get_macro_series([], D, D)
    with base as entered:
        assert entered is base


class _FakeSession:
    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        self.calls: list[dict] = []

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        resp = requests.Response()
        resp.status_code = self.statuses.pop(0)
        resp._content = b'{"ok": true}'
        resp.url = url
        return resp

    def close(self):
        return None


class _Http(BaseHTTPAdapter):
    name = "fakehttp"
    retry_min_wait_s = 0.0
    retry_max_wait_s = 0.0


def test_http_seam_retries_5xx_then_fails():
    session = _FakeSession([503, 502, 500])
    with pytest.raises(AdapterError, match="failed after 3 attempts"):
        _Http(session=session)._http("GET", "https://example.invalid/x")
    assert len(session.calls) == 3


def test_http_seam_recovers_after_a_transient_error():
    session = _FakeSession([503, 200])
    resp = _Http(session=session)._http("GET", "https://example.invalid/x", params={"a": 1})
    assert resp.status_code == 200 and resp.json() == {"ok": True}
    assert len(session.calls) == 2 and session.calls[0]["params"] == {"a": 1}
    assert session.calls[0]["headers"]["User-Agent"].startswith("quant-platform/")


def test_http_seam_4xx_raises_immediately():
    session = _FakeSession([404])
    with pytest.raises(AdapterError, match="HTTP 404"):
        _Http(session=session)._http("GET", "https://example.invalid/missing")
    assert len(session.calls) == 1


def test_adapter_result_helpers():
    empty = AdapterResult(vendor="v", frame=pd.DataFrame(), requested=["A", "B"], errors={"A": "x", "B": "y"})
    assert empty.ok == [] and empty.failed == ["A", "B"] and empty.all_failed
    partial = AdapterResult(
        vendor="v", frame=pd.DataFrame({"instrument_id": ["B", "B"]}), requested=["A", "B"], errors={"A": "x"}
    )
    assert partial.ok == ["B"] and not partial.all_failed and "1 failed out of 2" in partial.summary()
    assert Capability("bars_daily") is Capability.BARS_DAILY
