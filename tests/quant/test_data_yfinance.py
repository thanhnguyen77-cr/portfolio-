from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import fake_history, load_yf_fixture

from quant.core.errors import AdapterError
from quant.core.time import frozen_now
from quant.data.adapters.yfinance_adapter import YFinanceAdapter, default_history

START, END = date(2024, 7, 1), date(2024, 7, 12)


@pytest.fixture
def frames():
    return {"AAPL": load_yf_fixture("AAPL"), "VOD.L": load_yf_fixture("VOD.L")}


def _adapter(frames, **kw):
    return YFinanceAdapter(history_fn=fake_history(frames, **kw), drop_unfinished_session=False)


def test_us_fixture_parses_to_canonical_bars(frames, catalog):
    aapl = catalog["XNAS_AAPL"]
    res = _adapter(frames).get_bars([aapl], START, END)
    assert res.errors == {}
    bars = res.frame
    assert len(bars) == 9 and date(2024, 7, 4) not in set(bars["session_date"].dt.date)
    assert bars["close"].iloc[0] == pytest.approx(216.75)
    assert (bars["currency"] == "USD").all() and (bars["vendor"] == "yfinance").all()
    assert (bars["adjustment_state"] == "split_adjusted").all()
    closes = bars.set_index(bars["session_date"].dt.date)["ts_close_utc"]
    assert closes[date(2024, 7, 3)] == pd.Timestamp("2024-07-03 17:00", tz="UTC")  # early close
    assert closes[date(2024, 7, 5)] == pd.Timestamp("2024-07-05 20:00", tz="UTC")
    assert res.extras["corporate_actions"].empty


def test_uk_fixture_scales_pence_to_pounds_including_dividends(frames, catalog):
    vod = catalog["XLON_VOD"]
    res = _adapter(frames).get_bars([vod], START, END)
    bars = res.frame
    assert len(bars) == 10 and date(2024, 7, 4) in set(bars["session_date"].dt.date)
    assert bars["close"].iloc[0] == pytest.approx(0.7002)
    assert (bars["currency"] == "GBP").all()
    assert bars["ts_close_utc"].iloc[0] == pd.Timestamp("2024-07-01 15:30", tz="UTC")  # BST
    acts = res.extras["corporate_actions"]
    assert len(acts) == 1
    div = acts.iloc[0]
    assert div["action_type"] == "dividend" and div["value"] == pytest.approx(0.045)
    assert div["currency"] == "GBP" and pd.Timestamp(div["ex_date"]) == pd.Timestamp("2024-07-11")
    assert len(_adapter(frames).get_corporate_actions([vod], START, END).frame) == 1


def test_partial_failure_is_reported_not_raised(frames, catalog):
    res = _adapter(frames, fail={"VOD.L"}).get_bars([catalog["XNAS_AAPL"], catalog["XLON_VOD"]], START, END)
    assert res.ok == ["XNAS_AAPL"]
    assert res.failed == ["XLON_VOD"] and "simulated vendor failure" in res.errors["XLON_VOD"]
    assert not res.all_failed
    assert res.raise_if_all_failed() is res
    assert "1 failed out of 2" in res.summary()


def test_all_failed(frames, catalog):
    both = [catalog["XNAS_AAPL"], catalog["XLON_VOD"]]
    res = _adapter(frames, fail={"AAPL", "VOD.L"}).get_bars(both, START, END)
    assert res.all_failed
    with pytest.raises(AdapterError, match="every requested instrument failed"):
        res.raise_if_all_failed()


def test_empty_vendor_frame_is_an_error_entry(frames, catalog):
    res = _adapter(frames).get_bars([catalog["XNAS_AAPL"]], date(2030, 1, 1), date(2030, 1, 5))
    assert res.frame.empty and res.errors["XNAS_AAPL"].startswith("no data returned")


def test_unknown_symbol_is_an_error_entry(frames, catalog):
    res = _adapter(frames).get_bars([catalog["XNYS_JPM"]], START, END)
    assert "KeyError" in res.errors["XNYS_JPM"]


def test_unfinished_session_is_dropped_only_before_the_close(frames, catalog):
    aapl = catalog["XNAS_AAPL"]
    adapter = YFinanceAdapter(history_fn=fake_history(frames), drop_unfinished_session=True)
    with frozen_now("2024-07-12T15:00:00Z"):  # NYSE still open
        bars = adapter.get_bars([aapl], START, END).frame
    assert bars["session_date"].max().date() == date(2024, 7, 11)
    with frozen_now("2024-07-12T21:00:00Z"):  # after the 20:00 UTC close
        bars = adapter.get_bars([aapl], START, END).frame
    assert bars["session_date"].max().date() == date(2024, 7, 12)


def test_splits_are_parsed(frames, catalog):
    raw = frames["AAPL"].copy()
    raw.loc[raw.index[3], "Stock Splits"] = 4.0
    res = _adapter({"AAPL": raw}).get_bars([catalog["XNAS_AAPL"]], START, END)
    acts = res.extras["corporate_actions"]
    assert acts["action_type"].tolist() == ["split"] and acts["value"].iloc[0] == 4.0
    assert acts["currency"].iloc[0] is None


def test_seam_receives_vendor_symbol_and_range(frames, catalog):
    calls: list = []
    YFinanceAdapter(history_fn=fake_history(frames, calls=calls)).get_bars([catalog["XLON_VOD"]], START, END)
    assert calls == [("VOD.L", START, END, "1d")]


def test_intraday_is_rejected(frames, catalog):
    with pytest.raises(AdapterError, match="daily"):
        _adapter(frames).get_bars([catalog["XNAS_AAPL"]], START, END, freq="5m")


def test_default_seam_is_blocked_by_the_network_guard():
    with pytest.raises(RuntimeError, match="yfinance network call blocked"):
        default_history("AAPL", START, END, "1d")
