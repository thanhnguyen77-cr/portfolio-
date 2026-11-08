"""S13 adapters: Stooq, Alpaca data, Polygon, Tiingo, Nasdaq Data Link, Companies House, EDGAR filings,
licensed stubs (contract + guidance), and the crypto helpers. All HTTP goes through fake sessions."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests
from conftest import FIXTURES

from quant.core.calendars import get_calendar
from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.types import Instrument
from quant.crypto import (
    CRYPTO_CAVEATS,
    CRYPTO_PERIODS_PER_YEAR,
    align_to_calendar,
    crypto_pairs,
    realised_vol_crypto,
    weekend_share_of_variance,
)
from quant.data.adapters.alpaca_data import AlpacaDataAdapter
from quant.data.adapters.base import MarketDataAdapter
from quant.data.adapters.boe import BankOfEnglandAdapter
from quant.data.adapters.companies_house import CompaniesHouseAdapter
from quant.data.adapters.fred import FREDAdapter
from quant.data.adapters.nasdaq_data_link import NasdaqDataLinkAdapter, split_series_id
from quant.data.adapters.ons import ONSAdapter
from quant.data.adapters.polygon import PolygonAdapter
from quant.data.adapters.sec_edgar import SECEdgarAdapter
from quant.data.adapters.stooq import StooqAdapter
from quant.data.adapters.stubs import LICENSED_STUBS, BloombergAdapter, LicensedStub
from quant.data.adapters.tiingo import TiingoAdapter
from quant.data.adapters.yfinance_adapter import YFinanceAdapter
from quant.data.store import coerce_bars

START, END = date(2024, 7, 1), date(2024, 7, 12)


class RoutedSession:
    """Fake requests.Session: the first route whose substring matches the URL (or a param value) answers."""

    def __init__(self, routes: dict[str, tuple[int, str | bytes]]):
        self.routes = routes
        self.calls: list[dict] = []

    def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        haystack = url + " " + json.dumps(kw.get("params") or {}, default=str)
        for key, (status, body) in self.routes.items():
            if key in haystack:
                resp = requests.Response()
                resp.status_code = status
                resp._content = body if isinstance(body, bytes) else body.encode()
                resp.url = url
                return resp
        resp = requests.Response()
        resp.status_code = 404
        resp._content = b"no route"
        resp.url = url
        return resp

    def close(self):
        return None


def _fx(name: str) -> str:
    return (FIXTURES / name).read_text()


def _check_bars(bars: pd.DataFrame, vendor: str, state: str) -> None:
    out = coerce_bars(bars)  # canonical schema, no nulls, sorted
    assert (out["vendor"] == vendor).all() and (out["adjustment_state"] == state).all()
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out["ts_close_utc"].notna().all()


# ---------------------------------------------------------------- contract over every adapter
ALL_ADAPTERS = [
    lambda: YFinanceAdapter(history_fn=lambda *a: pd.DataFrame()),
    lambda: StooqAdapter(session=RoutedSession({})),
    lambda: FREDAdapter(session=RoutedSession({})),
    lambda: BankOfEnglandAdapter(session=RoutedSession({})),
    lambda: ONSAdapter(session=RoutedSession({})),
    lambda: SECEdgarAdapter(user_agent="t t@example.com", session=RoutedSession({})),
    lambda: AlpacaDataAdapter("k", "s", session=RoutedSession({})),
    lambda: PolygonAdapter("k", session=RoutedSession({})),
    lambda: TiingoAdapter("k", session=RoutedSession({})),
    lambda: NasdaqDataLinkAdapter(session=RoutedSession({})),
    lambda: CompaniesHouseAdapter("k", session=RoutedSession({})),
    *[lambda cls=cls: cls() for cls in LICENSED_STUBS],
]


@pytest.mark.parametrize("factory", ALL_ADAPTERS, ids=lambda f: f().name)
def test_every_adapter_honours_the_contract(factory, catalog):
    adapter = factory()
    assert isinstance(adapter, MarketDataAdapter)
    assert isinstance(adapter.name, str) and adapter.name
    caps = adapter.capabilities()
    assert isinstance(caps, frozenset)
    assert adapter.rate_limit is not None and adapter.field_map is not None
    with adapter as entered:
        assert entered is adapter
    inst = catalog["XNAS_AAPL"]
    from quant.data.adapters.base import Capability

    # every method either is a declared capability or raises an AdapterError (never silently succeeds)
    for cap, call in (
        (Capability.BARS_DAILY, lambda: adapter.get_bars([inst], START, END)),
        (Capability.CORPORATE_ACTIONS, lambda: adapter.get_corporate_actions([inst], START, END)),
        (Capability.MACRO, lambda: adapter.get_macro_series(["X"], START, END)),
        (Capability.FUNDAMENTALS, lambda: adapter.get_fundamentals([inst])),
        (Capability.EARNINGS, lambda: adapter.get_earnings([inst])),
    ):
        if cap in caps:
            continue
        with pytest.raises(AdapterError):
            call()
    if isinstance(adapter, LicensedStub):
        assert caps == frozenset() and adapter.intended_capabilities
        with pytest.raises(AdapterNotConfigured, match="not wired") as exc:
            adapter.get_bars([inst], START, END)
        assert adapter.install_hint.split()[0] in str(exc.value) and adapter.identifier_key in str(exc.value)


def test_licensed_stub_guidance_is_specific():
    bbg = BloombergAdapter()
    steps = bbg.wiring_steps()
    assert any("blpapi" in s for s in steps) and any("VOD LN Equity" in s for s in steps)
    assert bbg.field_hints["close"] == "PX_LAST" and not bbg.sdk_available()
    names = {cls().name for cls in LICENSED_STUBS}
    assert names == {"bloomberg", "refinitiv", "factset", "capiq"}
    keys = {cls().identifier_key for cls in LICENSED_STUBS}
    assert keys == {"bbg", "ric", "factset", "capiq"}


# ---------------------------------------------------------------- stooq
def test_stooq_parses_csv_and_reports_non_csv_bodies(catalog):
    session = RoutedSession(
        {"aapl.us": (200, _fx("stooq_aapl.csv")), "vod.uk": (200, "Exceeded the daily hits limit")}
    )
    adapter = StooqAdapter(session=session, drop_unfinished_session=False)
    res = adapter.get_bars([catalog["XNAS_AAPL"], catalog["XLON_VOD"]], START, END)
    assert res.ok == ["XNAS_AAPL"] and "no CSV" in res.errors["XLON_VOD"]
    bars = res.frame
    assert len(bars) == 5 and bars["close"].iloc[0] == pytest.approx(216.75)
    _check_bars(bars, "stooq", "split_adjusted")
    assert session.calls[0]["params"]["d1"] == "20240701" and session.calls[0]["params"]["s"] == "aapl.us"
    assert adapter.symbol_for(catalog["XLON_VOD"]) == "vod.uk"
    with pytest.raises(AdapterError, match="only daily"):
        adapter.get_bars([catalog["XNAS_AAPL"]], START, END, freq="1h")
    windowed = adapter.get_bars([catalog["XNAS_AAPL"]], date(2024, 7, 2), date(2024, 7, 3)).frame
    assert len(windowed) == 2


# ---------------------------------------------------------------- alpaca data
def test_alpaca_data_batches_symbols_and_needs_keys(catalog):
    session = RoutedSession({"stocks/bars": (200, _fx("alpaca_bars.json"))})
    adapter = AlpacaDataAdapter("key", "secret", session=session, drop_unfinished_session=False)
    insts = [catalog["XNAS_AAPL"], catalog["XNAS_MSFT"], catalog["XLON_VOD"]]
    res = adapter.get_bars(insts, START, END)
    assert res.ok == ["XNAS_AAPL", "XNAS_MSFT"] and "US equities only" in res.errors["XLON_VOD"]
    assert len(session.calls) == 1 and session.calls[0]["params"]["symbols"] == "AAPL,MSFT"
    assert (
        session.calls[0]["headers"]["APCA-API-KEY-ID"] == "key"
        and session.calls[0]["params"]["adjustment"] == "raw"
    )
    bars = res.frame
    aapl = bars[bars["instrument_id"] == "XNAS_AAPL"]
    assert aapl["session_date"].dt.date.tolist() == [date(2024, 7, 1), date(2024, 7, 2), date(2024, 7, 3)]
    _check_bars(bars, "alpaca_data", "raw")
    with pytest.raises(AdapterNotConfigured, match="QUANT_ALPACA"):
        AlpacaDataAdapter(None, None, session=session).get_bars([catalog["XNAS_AAPL"]], START, END)


# ---------------------------------------------------------------- polygon
def test_polygon_aggs_and_reference_actions(catalog):
    session = RoutedSession(
        {
            "aggs/ticker/AAPL": (200, _fx("polygon_aggs_aapl.json")),
            "reference/dividends": (200, _fx("polygon_dividends_aapl.json")),
            "reference/splits": (200, _fx("polygon_splits_aapl.json")),
        }
    )
    adapter = PolygonAdapter("key", session=session, drop_unfinished_session=False)
    adapter.rate_limit = adapter.rate_limit.__class__(min_interval_s=0.0)
    res = adapter.get_bars([catalog["XNAS_AAPL"]], START, END)
    assert res.errors == {} and len(res.frame) == 3
    assert res.frame["session_date"].dt.date.iloc[0] == date(2024, 7, 1)  # ms epoch -> New York session date
    _check_bars(res.frame, "polygon", "raw")
    assert session.calls[0]["params"]["apiKey"] == "key" and session.calls[0]["params"]["adjusted"] == "false"
    acts = adapter.get_corporate_actions([catalog["XNAS_AAPL"]], date(2020, 1, 1), END).frame
    kinds = acts.set_index("action_type")
    assert (
        kinds.loc["dividend", "value"] == pytest.approx(0.25) and kinds.loc["dividend", "currency"] == "USD"
    )
    assert kinds.loc["split", "value"] == pytest.approx(4.0) and pd.Timestamp(
        kinds.loc["split", "ex_date"]
    ) == pd.Timestamp("2020-08-31")
    with pytest.raises(AdapterNotConfigured, match="QUANT_POLYGON"):
        PolygonAdapter(None, session=session).get_bars([catalog["XNAS_AAPL"]], START, END)


# ---------------------------------------------------------------- tiingo
def test_tiingo_prices_and_embedded_actions(catalog):
    session = RoutedSession({"tiingo/daily/AAPL/prices": (200, _fx("tiingo_aapl.json"))})
    adapter = TiingoAdapter("tok", session=session, drop_unfinished_session=False)
    res = adapter.get_bars([catalog["XNAS_AAPL"]], START, END)
    assert res.errors == {} and len(res.frame) == 3 and res.frame["close"].iloc[-1] == pytest.approx(221.55)
    assert session.calls[0]["headers"]["Authorization"] == "Token tok"
    _check_bars(res.frame, "tiingo", "raw")
    acts = adapter.get_corporate_actions([catalog["XNAS_AAPL"]], START, END).frame.set_index("action_type")
    assert acts.loc["dividend", "value"] == pytest.approx(0.25) and pd.Timestamp(
        acts.loc["dividend", "ex_date"]
    ) == pd.Timestamp("2024-07-02")
    assert acts.loc["split", "value"] == pytest.approx(4.0)
    denied = TiingoAdapter("bad", session=RoutedSession({"prices": (200, '{"detail": "Invalid token."}')}))
    out = denied.get_bars([catalog["XNAS_AAPL"]], START, END)
    assert "Invalid token" in out.errors["XNAS_AAPL"]
    with pytest.raises(AdapterNotConfigured, match="QUANT_TIINGO"):
        TiingoAdapter(None, session=session).get_bars([catalog["XNAS_AAPL"]], START, END)


# ---------------------------------------------------------------- nasdaq data link
def test_ndl_dataset_columns_and_ids():
    assert split_series_id("LBMA/GOLD.2") == ("LBMA", "GOLD", "2")
    assert split_series_id("ml/emhyy") == ("ML", "EMHYY", None)
    with pytest.raises(AdapterError, match="DATABASE/DATASET"):
        split_series_id("GOLD")
    session = RoutedSession({"datasets/LBMA/GOLD/data.json": (200, _fx("ndl_dataset.json"))})
    adapter = NasdaqDataLinkAdapter(api_key="k", session=session)
    res = adapter.get_macro_series(
        ["LBMA/GOLD", "LBMA/GOLD.2", "LBMA/GOLD.GBP (PM)", "LBMA/GOLD.9", "FOO/BAR"], START, END
    )
    frame = res.frame.set_index("series_id")
    assert frame.loc["LBMA/GOLD", "value"].tolist() == pytest.approx(
        [2330.75, 2331.2, 2355.0]
    )  # first numeric column
    assert frame.loc["LBMA/GOLD.2", "value"].tolist() == pytest.approx([2328.35, 2360.1])  # null dropped
    assert frame.loc["LBMA/GOLD.GBP (PM)", "value"].tolist() == pytest.approx([1841.5, 1844.1, 1858.0])
    assert "out of range" in res.errors["LBMA/GOLD.9"] and "HTTP 404" in res.errors["FOO/BAR"]
    assert session.calls[0]["params"]["api_key"] == "k"
    assert (frame["vendor"] == "ndl").all()


# ---------------------------------------------------------------- companies house + EDGAR filings
def test_companies_house_filings_index(catalog):
    vod = catalog["XLON_VOD"].model_copy(
        update={"identifiers": {**catalog["XLON_VOD"].identifiers, "companies_house": "1833679"}}
    )
    session = RoutedSession({"/company/01833679/filing-history": (200, _fx("companies_house_filings.json"))})
    adapter = CompaniesHouseAdapter("key", session=session)
    res = adapter.get_filings([vod, catalog["XLON_BP"]], since=date(2023, 6, 1))
    assert res.ok == ["XLON_VOD"] and "companies_house" in res.errors["XLON_BP"]
    frame = res.frame
    assert len(frame) == 2 and frame["form"].tolist() == ["CS01", "AA"]
    assert frame["filed_at"].dt.date.tolist() == [date(2023, 12, 1), date(2024, 5, 14)]
    assert frame["period_end"].dt.date.iloc[-1] == date(2024, 3, 31) and frame["url"].iloc[-1].startswith(
        "https://"
    )
    assert (
        session.calls[0]["auth"] == ("key", "")
        and session.calls[0]["params"]["category"] == "accounts,confirmation-statement"
    )
    with pytest.raises(AdapterNotConfigured, match="QUANT_COMPANIES_HOUSE"):
        CompaniesHouseAdapter(None, session=session).get_filings([vod])


def test_edgar_filings_index_from_submissions(catalog):
    aapl = catalog["XNAS_AAPL"].model_copy(
        update={"identifiers": {**catalog["XNAS_AAPL"].identifiers, "cik": "320193"}}
    )
    session = RoutedSession({"submissions/CIK0000320193.json": (200, _fx("edgar_submissions_sample.json"))})
    adapter = SECEdgarAdapter(user_agent="Jane Doe jane@example.com", session=session)
    res = adapter.get_filings([aapl], forms=("10-Q",), since=date(2024, 3, 1))
    frame = res.frame
    assert res.errors == {} and frame["form"].tolist() == ["10-Q", "10-Q"]
    assert frame["filed_at"].dt.date.tolist() == [date(2024, 5, 3), date(2024, 8, 2)]
    assert frame["period_end"].dt.date.tolist() == [date(2024, 3, 30), date(2024, 6, 29)]
    assert (
        frame["url"].iloc[0]
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019324000069/aapl-20240330.htm"
    )
    assert (frame["category"] == "financial").all()
    everything = adapter.get_filings([aapl], forms=()).frame
    assert len(everything) == 4 and "event" in set(everything["category"])
    with pytest.raises(AdapterNotConfigured):
        SECEdgarAdapter(user_agent=None, session=session).get_filings([aapl])


# ---------------------------------------------------------------- crypto
def test_crypto_calendar_pairs_and_alignment(catalog):
    pairs = crypto_pairs(catalog)
    assert {p.symbol for p in pairs} >= {"BTCUSD", "ETHUSD"} and all(
        p.calendar_code == "CRYPTO" for p in pairs
    )
    cal = get_calendar("CRYPTO")
    sessions = cal.sessions(date(2024, 12, 20), date(2025, 1, 5))
    assert len(sessions) == 17  # every day, Christmas and weekends included
    assert cal.is_session(date(2024, 12, 25)) and cal.session_close(date(2024, 1, 6)).hour == 23
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120))), index=idx, name="BTCUSD")
    aligned = align_to_calendar(px, "XNYS")
    assert all(get_calendar("XNYS").is_session(d) for d in aligned.index.date)
    monday = aligned.index[aligned.index.day_of_week == 0][2]
    assert aligned[monday] == px[monday]  # last price at or before the session date
    friday = aligned.index[aligned.index.get_loc(monday) - 1]
    assert np.log(aligned[monday] / aligned[friday]) == pytest.approx(np.log(px[monday] / px[friday]))
    share = weekend_share_of_variance(px)
    assert 0.1 < share < 0.5  # ~2/7 of days are weekend days
    rv = realised_vol_crypto(px, window=30).dropna()
    assert len(rv) == 120 - 30 and rv.iloc[-1] == pytest.approx(
        np.log(px).diff().iloc[-30:].std(ddof=1) * np.sqrt(CRYPTO_PERIODS_PER_YEAR)
    )
    assert CRYPTO_PERIODS_PER_YEAR == 365 and any("24/7" in c for c in CRYPTO_CAVEATS)
    assert Instrument is not None
