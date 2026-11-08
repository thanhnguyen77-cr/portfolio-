from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES, FakeMacroAdapter, make_bars

from quant.config.settings import load_settings
from quant.core.errors import ConfigError
from quant.data.adapters.base import Capability
from quant.data.adapters.ons import ONSAdapter, parse_ons_date
from quant.data.service import DataService
from quant.macro import (
    MACRO_SERIES,
    annualised_3m,
    breakeven,
    cross_asset_dashboard,
    curvature,
    curve_snapshot,
    growth_inflation_quadrant,
    inflation_regime,
    inversion_flags,
    macro_series_spec,
    mom,
    real_rate,
    slope,
    yoy,
)
from quant.macro.catalog import series_for_region


def test_ons_dates_and_payload_parsing():
    assert parse_ons_date("2024 JAN") == pd.Timestamp("2024-01-01") and parse_ons_date(
        "2024 Q3"
    ) == pd.Timestamp("2024-07-01")
    assert parse_ons_date("2023") == pd.Timestamp("2023-01-01") and parse_ons_date("garbage") is None
    payload = json.loads((FIXTURES / "ons_timeseries_sample.json").read_text())
    df = ONSAdapter.parse_payload("MM23/L55O", payload)
    assert (
        len(df) == 6
        and df["value"].tolist()[:2] == [4.2, 3.8]
        and df["date"].iloc[0] == pd.Timestamp("2024-01-01")
    )
    quarterly = ONSAdapter.parse_payload("X/Y", {"quarters": payload["quarters"], "years": payload["years"]})
    assert len(quarterly) == 2 and quarterly["date"].iloc[1] == pd.Timestamp("2024-04-01")
    assert ONSAdapter.parse_payload("X/Y", {}).empty
    assert ONSAdapter.split_id("mm23/l55o") == ("MM23", "L55O")
    with pytest.raises(ValueError):
        ONSAdapter.split_id("L55O")


def test_ons_adapter_through_the_seam():
    payload = json.loads((FIXTURES / "ons_timeseries_sample.json").read_text())
    adapter = ONSAdapter()
    urls: list[str] = []

    class _Resp:
        def json(self):
            return payload

    def _http(method, url, **kw):
        urls.append(url)
        return _Resp()

    adapter._http = _http
    res = adapter.get_macro_series(["MM23/L55O", "BAD"], date(2024, 2, 1), date(2024, 5, 31))
    assert (
        res.frame["value"].tolist() == [3.8, 3.8, 3.0, 2.8] and (res.frame["series_id"] == "MM23/L55O").all()
    )
    assert "ValueError" in res.errors["BAD"] and urls == [
        "https://api.ons.gov.uk/timeseries/L55O/dataset/MM23/data"
    ]
    assert adapter.get_macro_series(["MM23/L55O"], date(2030, 1, 1), date(2030, 2, 1)).errors


def test_catalog_integrity():
    assert all(s.vendor in {"fred", "boe", "ons"} for s in MACRO_SERIES.values())
    assert all(name == s.name for name, s in MACRO_SERIES.items())
    assert all("/" in s.series_id for s in MACRO_SERIES.values() if s.vendor == "ons")
    assert macro_series_spec("us_cpi").series_id == "CPIAUCSL"
    unverified = [s.name for s in MACRO_SERIES.values() if not s.verified]
    assert "uk_10y" in unverified and "us_10y" not in unverified
    assert len(series_for_region("us")) + len(series_for_region("uk")) == len(series_for_region("both"))
    with pytest.raises(KeyError):
        macro_series_spec("nope")


def test_yield_curve_slope_inversion_and_snapshot():
    idx = pd.bdate_range("2024-01-01", periods=200)
    y10 = pd.Series(4.0, index=idx)
    y2 = pd.Series(np.r_[np.full(100, 4.5), np.full(100, 3.5)], index=idx)
    sp = slope(y10, y2)
    flags = inversion_flags(sp)
    assert flags["inverted"].iloc[:100].all() and not flags["inverted"].iloc[100:].any()
    assert flags["sessions_inverted"].iloc[99] == 100 and flags["sessions_inverted"].iloc[100] == 0
    assert flags["uninversion_event"].sum() == 1 and flags.index[flags["uninversion_event"]][0] == idx[100]
    y5 = pd.Series(4.4, index=idx)
    assert curvature(y2, y5, y10).iloc[-1] == pytest.approx(2 * 4.4 - 3.5 - 4.0)
    snap = curve_snapshot({"2y": y2, "10y": y10}).set_index("tenor")
    assert snap.loc["2y", "level"] == 3.5 and snap.loc["2y", "chg_3m_bp"] == 0.0  # step older than 3 months
    assert snap.loc["10y", "chg_1m_bp"] == 0.0
    recent = pd.Series(np.r_[np.full(170, 4.5), np.full(30, 3.5)], index=idx)  # step ~6 weeks before the end
    snap2 = curve_snapshot({"2y": recent}).set_index("tenor")
    assert snap2.loc["2y", "chg_3m_bp"] == pytest.approx(-100.0) and snap2.loc["2y", "chg_1m_bp"] == 0.0
    assert curve_snapshot({"2y": recent}, as_of=idx[150]).set_index("tenor").loc["2y", "level"] == 4.5
    sparse = slope(y10, y2.iloc[::5])  # short leg sparser: forward-filled onto the union
    assert len(sparse) == 200 and sparse.notna().all()


def test_inflation_transforms():
    idx = pd.date_range("2020-01-01", periods=48, freq="MS")
    cpi = pd.Series(100 * 1.0025 ** np.arange(48), index=idx)
    assert yoy(cpi).iloc[-1] == pytest.approx((1.0025**12 - 1) * 100)
    assert mom(cpi).iloc[-1] == pytest.approx(0.25)
    assert annualised_3m(cpi).iloc[-1] == pytest.approx((1.0025**12 - 1) * 100)
    nominal = pd.Series(5.0, index=pd.bdate_range("2023-01-02", periods=300))
    rr = real_rate(nominal, yoy(cpi))
    assert rr.iloc[-1] == pytest.approx(5.0 - yoy(cpi).dropna().iloc[-1], abs=0.02)
    be = breakeven(nominal, pd.Series(2.2, index=nominal.index[::7]))
    assert be.iloc[-1] == pytest.approx(2.8)
    rising = pd.Series(np.linspace(0.5, 5.0, 24), index=idx[:24])
    reg = inflation_regime(rising)
    assert reg["regime"].iloc[-1] == "above_target_rising" and reg["regime"].iloc[3] == "below_target_rising"
    assert inflation_regime(pd.Series(2.0, index=idx[:12]))["regime"].iloc[-1] == "at_target_stable"


def test_dashboard_quadrant_and_signals():
    idx = pd.bdate_range("2019-01-01", periods=1500)
    rising = pd.Series(np.linspace(0, 5, 1500), index=idx)
    falling = pd.Series(np.linspace(5, 0, 1500), index=idx)
    board = cross_asset_dashboard(
        {"growth_ip_yoy": rising, "inflation_yoy": falling, "credit_hy_oas": falling, "vol_index": rising}
    )
    t = board["table"].set_index("indicator")
    assert board["quadrant"] == "goldilocks / disinflationary growth"
    assert t.loc["growth_ip_yoy", "signal"] == "risk_on" and t.loc["credit_hy_oas", "signal"] == "risk_on"
    assert t.loc["vol_index", "signal"] == "risk_off" and t.loc["inflation_yoy", "signal"] == "neutral"
    assert "rates_10y" in board["missing"] and "growth_ip_yoy" not in board["missing"]
    assert t.loc["growth_ip_yoy", "chg_3m"] > 0 and np.isfinite(t.loc["growth_ip_yoy", "z_5y"])
    assert growth_inflation_quadrant(-1.0, 1.0) == "stagflation" and growth_inflation_quadrant(
        np.nan, 1.0
    ).startswith("unknown")
    assert growth_inflation_quadrant(0.0, 0.0) == "reflation / overheating"  # flat counts as up by convention
    empty = cross_asset_dashboard({})
    assert empty["table"].empty and empty["quadrant"].startswith("unknown")
    unemployment_only = cross_asset_dashboard({"growth_unemployment": rising, "inflation_yoy": rising})
    assert unemployment_only["quadrant"] == "stagflation"  # rising unemployment = growth down


def test_service_macro_series_and_dashboard(tmp_path, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(paths={"data_dir": str(tmp_path / "data")})
    idx_m = pd.date_range("2018-01-01", "2024-06-01", freq="MS")
    idx_d = pd.bdate_range("2018-01-01", "2024-06-28")
    fred = FakeMacroAdapter(
        {
            "CPIAUCSL": pd.Series(250 * 1.0025 ** np.arange(len(idx_m)), index=idx_m),
            "INDPRO": pd.Series(100 + np.linspace(0, 10, len(idx_m)), index=idx_m),
            "UNRATE": pd.Series(np.linspace(4.0, 3.5, len(idx_m)), index=idx_m),
            "DGS10": pd.Series(np.linspace(2.5, 4.5, len(idx_d)), index=idx_d),
            "DGS2": pd.Series(np.linspace(2.0, 4.8, len(idx_d)), index=idx_d),
            "DFF": pd.Series(np.linspace(1.5, 5.3, len(idx_d)), index=idx_d),
            "BAMLH0A0HYM2": pd.Series(3.5, index=idx_d),
        },
        name="fred",
    )
    svc = DataService(settings, catalog=catalog, adapters={"fred": fred})
    cpi = svc.macro_series("us_cpi", date(2018, 1, 1), date(2024, 6, 28))
    assert cpi.name == "us_cpi" and len(cpi) == len(idx_m)
    with pytest.raises(ConfigError, match="unknown macro series"):
        svc.macro_series("nope", date(2018, 1, 1))
    for ref, seed in (("XNYS_SPX", 1), ("XNYS_VIX", 2)):
        svc.parquet.write_bars(
            make_bars(catalog[ref], date(2018, 1, 2), date(2024, 6, 28), seed=seed, vendor="fred")
        )
    svc.settings.data.adapter_priority[:] = ["fred"]
    base_caps = fred.capabilities()  # let the macro fake also "serve" the cached index bars written above
    fred.capabilities = lambda: frozenset({*base_caps, Capability.BARS_DAILY})
    board = svc.macro_dashboard("us", end=date(2024, 6, 28), refresh=None)
    t = board["table"].set_index("indicator")
    assert {"growth_ip_yoy", "inflation_yoy", "rates_slope_10y_2y", "credit_hy_oas", "rates_policy"} <= set(
        t.index
    )
    assert t.loc["rates_slope_10y_2y", "value"] == pytest.approx(4.5 - 4.8, abs=0.05)
    assert "us_real_10y" in " ".join(board["notes"]) and "fx_dollar" in board["missing"]
    assert board["quadrant"] in {"reflation / overheating", "goldilocks / disinflationary growth"}
    with pytest.raises(ConfigError):
        svc.macro_dashboard("eu")
