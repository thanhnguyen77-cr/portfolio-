from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from conftest import FIXTURES, make_bars, plant

from quant.config.settings import load_settings
from quant.data.fx import FXRates
from quant.data.service import DataService
from quant.data.validation import validate_bars
from quant.reports import Section, render_report
from quant.universe.screen import ScreenConfig, screen_universe

SP = (FIXTURES / "wiki_sp500.html").read_text()
FTSE = (FIXTURES / "wiki_ftse100.html").read_text()
S, E = date(2024, 1, 2), date(2024, 12, 31)


@pytest.fixture
def svc(tmp_path, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(paths={"data_dir": str(tmp_path / "data")})
    return DataService(settings, catalog=catalog, adapters={})


def test_build_sp500_from_html_registers_reconstructs_and_records(svc):
    uni, info = svc.build_universe(
        "sp500", html=SP, as_of=date(2024, 7, 1), expected_rows=(3, 20), renames={"FB": "META"}
    )
    assert (
        info["members"] == 6 and info["quality"] == "reconstructed" and info["reconstructed_intervals"] == 8
    )
    assert set(uni.membership_at(date(2024, 7, 1))) == {
        "XNAS_AAPL",
        "XNAS_MSFT",
        "XNYS_BRK_B",
        "XNYS_JPM",
        "XNYS_XOM",
        "XNYS_META",
    }
    early = set(uni.membership_at(date(2024, 1, 4)))
    assert (
        {"XNYS_OLD1", "XNYS_OLD2", "XNYS_META"} <= early
        and "XNYS_JPM" not in early
        and "XNYS_XOM" not in early
    )
    assert set(uni.membership_at(date(2024, 3, 18))) == {
        "XNAS_AAPL",
        "XNAS_MSFT",
        "XNYS_BRK_B",
        "XNYS_JPM",
        "XNYS_XOM",
        "XNYS_META",
    }
    assert svc.catalog["XNYS_OLD1"].notes.startswith("former index member")
    iv = uni.intervals()
    assert (iv["instrument_id"] == "XNYS_META").sum() == 1  # FB -> META merged into one unbroken stint
    assert svc.db.lookup_alias("yahoo", "FB") == "XNYS_META"
    assert svc.lineage.recent()["operation"].tolist() == ["universe"]
    assert svc.db.get_instrument("XNYS_BRK_B") is not None
    assert uni.coverage(
        date(2024, 1, 1), date(2024, 7, 1), with_data={"XNAS_AAPL", "XNAS_MSFT"}
    ) == pytest.approx(2 / 8)


def test_build_ftse_is_current_only(svc):
    uni, info = svc.build_universe("ftse100", html=FTSE, as_of=date(2024, 7, 1), expected_rows=(3, 20))
    assert info["quality"] == "current_only" and any("current_only" in n for n in info["notes"])
    assert uni.membership_at(date(2024, 7, 1)) == ["XLON_BP", "XLON_BT_A", "XLON_RR", "XLON_SHEL", "XLON_VOD"]
    assert svc.catalog["XLON_BT_A"].price_scale == 0.01
    uni2, info2 = svc.build_universe("ftse100", html=FTSE, as_of=date(2024, 8, 1), expected_rows=(3, 20))
    assert info2["added"] == [] and info2["removed"] == [] and len(uni2.snapshots()) == 2


def _bars(catalog):
    aapl = make_bars(catalog["XNAS_AAPL"], S, E, seed=1)
    jpm = make_bars(catalog["XNYS_JPM"], S, E, seed=2)
    jpm["volume"] = jpm["volume"] * 1e-4  # illiquid
    xom = make_bars(catalog["XNYS_XOM"], S, E, seed=3, vol=0.06)  # ~95% annualised: outside the band
    vod = make_bars(catalog["XLON_VOD"], S, E, seed=4, start_price=0.7)
    vod["volume"] = vod["volume"] * 20
    gappy = plant(
        make_bars(catalog["XNAS_MSFT"], S, E, seed=5), drop_dates=[date(2024, 3, d) for d in range(4, 30)]
    )
    return pd.concat([aapl, jpm, xom, vod, gappy])


def test_screen_ranks_by_evidence(catalog):
    bars = _bars(catalog)
    fx = FXRates({"GBPUSD": pd.Series(1.27, index=pd.bdate_range(S, E))})
    dq = validate_bars(bars, catalog)
    table = screen_universe(bars, catalog, base_currency="USD", fx=fx, data_quality=dq)
    by = table.set_index("instrument_id")
    assert table["instrument_id"].iloc[0] == "XNAS_AAPL" and by.loc["XNAS_AAPL", "passes"]
    assert by.loc["XLON_VOD", "passes"] and by.loc["XLON_VOD", "currency"] == "GBP"
    assert by.loc["XLON_VOD", "adv_base"] == pytest.approx(by.loc["XLON_VOD", "adv_base"])  # finite
    assert not by.loc["XNYS_JPM", "passes"] and "adv" in by.loc["XNYS_JPM", "reasons"]
    assert not by.loc["XNYS_XOM", "passes"] and "vol" in by.loc["XNYS_XOM", "reasons"]
    assert (
        by.loc["XNAS_MSFT", "dq_grade"] == "C" and by.loc["XNAS_MSFT", "passes"]
    )  # C is acceptable by default
    stricter = screen_universe(
        bars, catalog, base_currency="USD", fx=fx, data_quality=dq, config=ScreenConfig(min_dq_grade="B")
    )
    assert "data quality" in stricter.set_index("instrument_id").loc["XNAS_MSFT", "reasons"]
    pts = table[["liquidity_pts", "quality_pts", "volatility_pts", "price_pts"]].sum(axis=1)
    assert (pts.round(1) - table["score"]).abs().max() < 0.11
    assert table["passes"].tolist() == sorted(table["passes"].tolist(), reverse=True)
    with pytest.raises(ValueError, match="FX rates needed"):
        screen_universe(bars, catalog, base_currency="USD", fx=None)
    assert screen_universe(bars.iloc[0:0], catalog).empty
    no_dq = screen_universe(bars[bars["instrument_id"] == "XNAS_AAPL"], catalog, base_currency="USD")
    assert no_dq["quality_pts"].iloc[0] == pytest.approx(15.0)  # half credit without a validation report
    strict = ScreenConfig(min_adv_base=1e12)
    assert not screen_universe(bars, catalog, base_currency="USD", fx=fx, config=strict)["passes"].any()


def test_service_screen_over_cache(svc, catalog):
    bars = make_bars(catalog["XNAS_AAPL"], S, E, seed=7, vendor="fake")
    svc.parquet.write_bars(bars)
    svc.settings.data.adapter_priority[:] = ["fake"]
    from conftest import FakeBarsAdapter

    svc.adapters["fake"] = FakeBarsAdapter({})
    table = svc.screen(["AAPL"], S, E, refresh=False)
    assert (
        len(table) == 1 and table["instrument_id"].iloc[0] == "XNAS_AAPL" and table["dq_grade"].iloc[0] == "A"
    )


def test_render_report(tmp_path):
    html = render_report(
        "Demo report",
        [
            Section("Text", text="hello", kv={"members": 5}),
            Section("Table", table=pd.DataFrame({"a": [1.23456], "b": ["x"]})),
        ],
        ["first caveat"],
        subtitle="sub",
        path=tmp_path / "r.html",
    )
    assert (
        "<title>Demo report</title>" in html
        and "hello" in html
        and "1.235" in html
        and "first caveat" in html
    )
    assert "Caveats" in html and "<script" not in html and (tmp_path / "r.html").exists()
    assert "general limitations" in render_report("Empty", [], [])
