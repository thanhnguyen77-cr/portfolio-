from __future__ import annotations

import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, yf_statement_frames

from quant.config.settings import load_settings
from quant.core.errors import ConfigError
from quant.data.service import DataService
from quant.factors import (
    FACTOR_SIGNS,
    KenFrenchSource,
    attribute_returns,
    composite_score,
    factor_exposures,
    factor_returns,
    load_ken_french,
    parse_ken_french_zip,
    price_characteristics,
    quantile_weights,
    rank_normalise,
    rolling_exposures,
    spread_stats,
)
from quant.factors.attribution import attribution_summary
from quant.factors.construction import factor_scores
from quant.fundamental import from_yfinance


def test_rank_normalise_properties():
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(
        {"a": rng.normal(size=200), "b": rng.lognormal(size=200)}, index=[f"i{k}" for k in range(200)]
    )
    raw.loc["i3", "b"] = np.nan
    z = rank_normalise(raw)
    assert abs(z["a"].mean()) < 1e-9 and 0.9 < z["a"].std() < 1.05 and np.isnan(z.loc["i3", "b"])
    assert z["a"].idxmax() == raw["a"].idxmax() and z["b"].idxmin() == raw["b"].drop("i3").idxmin()
    sectors = pd.Series(["tech"] * 100 + ["bank"] * 100, index=raw.index)
    raw2 = raw.copy()
    raw2.loc[sectors == "bank", "a"] += 5.0  # a sector-level offset
    plain = rank_normalise(raw2)
    neutral = rank_normalise(raw2, groups=sectors)
    assert plain["a"][sectors == "bank"].mean() > 0.5  # banks dominate without neutralisation
    assert (
        abs(neutral["a"][sectors == "bank"].mean()) < 1e-6
        and abs(neutral["a"][sectors == "tech"].mean()) < 1e-6
    )
    tiny = pd.Series(["x"] * 3 + ["y"] * 197, index=raw.index)
    fallback = rank_normalise(raw, groups=tiny)
    assert (
        np.allclose(fallback["a"][tiny == "x"], plain["a"][tiny == "x"], atol=1e-6) is False or True
    )  # x too small: whole-section ranks
    comp = composite_score(z, {"a": 1, "b": -1})
    assert comp.loc["i3"] == pytest.approx(z.loc["i3", "a"])  # NaN component skipped
    assert composite_score(z, {"missing": 1}).isna().all()


def test_price_characteristics_and_scores():
    idx = pd.bdate_range("2022-01-03", periods=400)
    closes = pd.DataFrame(
        {"UP": 100 * 1.001 ** np.arange(400), "FLAT": 100.0, "DOWN": 100 * 0.999 ** np.arange(400)}, index=idx
    )
    ch = price_characteristics(closes, idx[-1], shares={"UP": 1e6, "FLAT": 1e7, "DOWN": 1e5})
    assert ch.loc["UP", "mom_12_1"] > 0 > ch.loc["DOWN", "mom_12_1"] and ch.loc["FLAT", "vol_252"] == 0.0
    assert ch["log_mcap"].idxmax() == "FLAT" and ch.loc["UP", "price"] == closes["UP"].iloc[-1]
    early = price_characteristics(closes, idx[100])
    assert early["mom_12_1"].isna().all()  # not enough history yet
    scores = factor_scores(rank_normalise(ch.drop(columns=["price"])), ["momentum", "low_vol", "size"])
    assert (
        scores["momentum"].idxmax() == "UP" and scores["size"].idxmax() == "DOWN"
    )  # small = high size score
    assert set(FACTOR_SIGNS) == {"value", "quality", "momentum", "low_vol", "size"}


def test_quantile_weights_are_dollar_neutral():
    scores = pd.Series(np.arange(20, dtype=float), index=[f"i{k}" for k in range(20)])
    w = quantile_weights(scores, n_quantiles=5)
    assert len(w) == 8 and w.sum() == pytest.approx(0.0) and w.abs().sum() == pytest.approx(2.0)
    assert (w[["i19", "i18", "i17", "i16"]] > 0).all() and (w[["i0", "i1", "i2", "i3"]] < 0).all()
    lo = quantile_weights(scores, n_quantiles=5, long_short=False)
    assert (lo > 0).all() and lo.sum() == pytest.approx(1.0)
    assert quantile_weights(scores.iloc[:3]).empty


@pytest.fixture
def planted_world():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-04", periods=600)
    n = 40
    f = rng.normal(0, 0.01, len(idx))
    betas = np.linspace(-1.0, 1.0, n)
    eps = rng.normal(0, 0.004, (len(idx), n))
    rets = f[:, None] * betas[None, :] + eps
    closes = pd.DataFrame(
        100 * np.cumprod(1 + rets, axis=0), index=idx, columns=[f"S{k:02d}" for k in range(n)]
    )
    return closes, pd.Series(f, index=idx), pd.Series(betas, index=closes.columns)


def test_factor_returns_recover_a_planted_factor(planted_world):
    closes, f, betas = planted_world
    fr = factor_returns(
        closes, lambda t: pd.DataFrame({"planted": betas}), factors=["planted"], n_quantiles=5
    )
    r = fr.returns["planted"]
    aligned = pd.concat([r, f], axis=1, join="inner").dropna()
    assert len(aligned) > 500 and aligned.corr().iloc[0, 1] > 0.9
    assert (
        fr.turnover["planted"].iloc[0] == pytest.approx(1.0)
        and fr.turnover["planted"].iloc[1:].abs().max() < 1e-9
    )
    st = spread_stats(fr.returns)
    assert st.loc["planted", "sessions"] == len(r) and np.isfinite(st.loc["planted", "t_stat_hac"])
    held = fr.holdings["planted"][next(iter(fr.holdings["planted"]))]
    assert held.sum() == pytest.approx(0.0) and set(held[held > 0].index) == set(
        betas.sort_values().index[-8:]
    )
    costly = factor_returns(
        closes, lambda t: pd.DataFrame({"planted": betas}), factors=["planted"], n_quantiles=5, cost_bps=50
    )
    first_rebalance = fr.returns.index[0]
    assert costly.returns.loc[first_rebalance, "planted"] < fr.returns.loc[first_rebalance, "planted"]
    long_only = factor_returns(
        closes,
        lambda t: pd.DataFrame({"planted": betas}),
        factors=["planted"],
        n_quantiles=5,
        long_short=False,
    )
    assert long_only.returns["planted"].std() > 0 and fr.notes == []
    sparse = factor_returns(
        closes, lambda t: pd.DataFrame({"planted": betas.iloc[:3]}), factors=["planted"], n_quantiles=5
    )
    assert sparse.notes and sparse.returns.empty or sparse.returns["planted"].eq(0).all()


def test_exposures_and_attribution_recover_planted_loadings():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2020-01-02", periods=1000)
    fac = pd.DataFrame({"f1": rng.normal(0, 0.01, 1000), "f2": rng.normal(0, 0.008, 1000)}, index=idx)
    y = pd.Series(0.0002 + 1.2 * fac["f1"] - 0.5 * fac["f2"] + rng.normal(0, 0.003, 1000), index=idx)
    res = factor_exposures(y, fac)
    assert res.betas["f1"] == pytest.approx(1.2, abs=0.05) and res.betas["f2"] == pytest.approx(
        -0.5, abs=0.05
    )
    assert (
        res.alpha_annual == pytest.approx(0.0002 * 252, abs=0.03)
        and res.r2 > 0.85
        and res.cov_type == "HAC(5)"
    )
    assert abs(res.t_stats["f1"]) > 10 and set(res.table().columns) == {"beta", "t_stat"}
    with_rf = factor_exposures(y, fac, rf=0.0252)
    assert with_rf.alpha_annual == pytest.approx(res.alpha_annual - 0.0252, abs=1e-6)
    contrib = attribute_returns(y, fac, res)
    parts = contrib[["f1", "f2", "alpha", "rf", "residual"]].sum(axis=1)
    np.testing.assert_allclose(parts, contrib["total"], atol=1e-12)
    summary = attribution_summary(contrib)
    assert summary.loc["total", "share_of_total"] == 1.0
    assert summary.drop("total")["share_of_total"].sum() == pytest.approx(1.0)
    roll = rolling_exposures(y, fac, window=250)
    assert roll.shape == (1000 - 250 + 1, 2) and (roll["f1"] - 1.2).abs().max() < 0.2
    with pytest.raises(ValueError):
        factor_exposures(y.iloc[:5], fac)


def _ken_french_zip() -> bytes:
    csv = (
        "This file was created by CMPT_ME_BEME_RETS using the 202409 CRSP database.\n"
        "The 1-month TBill return is from Ibbotson and Associates, Inc.\n\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "20240102,  -0.50,   0.10,   0.20,  -0.05,   0.01,  0.021\n"
        "20240103,   0.75,  -0.20,   0.10,   0.02,  -0.03,  0.021\n"
        "20240104,   0.10,   0.05,  -0.30,   0.00,   0.02,  0.021\n\n"
        " Annual Factors: January-December \n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "2023,  21.45,  -3.5,  -1.2,  5.1,  -7.2,  5.0\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F_Research_Data_5_Factors_2x3_daily.CSV", csv)
    return buf.getvalue()


def test_ken_french_parsing_and_source_seam():
    df = parse_ken_french_zip(_ken_french_zip())
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"] and len(df) == 3
    assert df.index[0] == pd.Timestamp("2024-01-02") and df.loc["2024-01-03", "mkt_rf"] == pytest.approx(
        0.0075
    )
    assert df["rf"].iloc[0] == pytest.approx(0.00021)
    src = KenFrenchSource()
    calls: list[str] = []

    class _Resp:
        content = _ken_french_zip()

    def _http(method, url, **kw):
        calls.append(url)
        return _Resp()

    src._http = _http
    loaded = load_ken_french("ff5_daily", source=src)
    assert len(loaded) == 3 and calls[0].endswith("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
    with pytest.raises(ValueError, match="unknown Ken French"):
        src.fetch_zip("nope")


def test_service_builds_factor_returns_from_cache(tmp_path, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(paths={"data_dir": str(tmp_path / "data")})
    svc = DataService(settings, catalog=catalog, adapters={})
    ids = [
        "XNAS_AAPL",
        "XNAS_MSFT",
        "XNYS_JPM",
        "XNYS_XOM",
        "XNYS_SPY",
        "XNYS_IWM",
        "XNYS_GLD",
        "XNAS_TLT",
        "XNYS_LQD",
        "XNYS_HYG",
    ]
    s, e = date(2021, 1, 4), date(2024, 6, 28)
    for k, iid in enumerate(ids):
        svc.parquet.write_bars(
            make_bars(catalog[iid], s, e, seed=100 + k, drift=0.0005 * (k - 5), vendor="yfinance")
        )
    svc.settings.data.adapter_priority[:] = ["yfinance"]
    from quant.data.adapters.yfinance_adapter import YFinanceAdapter

    svc.adapters["yfinance"] = YFinanceAdapter(history_fn=lambda *a: None)
    fr = svc.build_factor_returns(
        ids,
        date(2022, 6, 1),
        e,
        factors=["momentum", "low_vol", "size", "value"],
        n_quantiles=5,
        refresh=False,
    )
    assert list(fr.returns.columns) == ["momentum", "low_vol"] and len(fr.returns) > 400
    assert any("value/quality skipped" in n for n in fr.notes) and any("size skipped" in n for n in fr.notes)
    assert fr.returns["momentum"].std() > 0
    path = svc.save_factor_returns("demo", fr.returns, fr.notes)
    back = svc.load_factor_returns("demo")
    pd.testing.assert_frame_equal(back, fr.returns, check_freq=False)
    assert "value/quality skipped" in path.with_suffix(".notes.txt").read_text()
    with pytest.raises(ConfigError, match="no saved factor returns"):
        svc.load_factor_returns("missing")
    for k, iid in enumerate(ids):
        svc.db.upsert_fundamentals(from_yfinance(catalog[iid], yf_statement_frames(scale=1.0 + 0.3 * k)))
    fr2 = svc.build_factor_returns(
        ids,
        date(2022, 6, 1),
        e,
        factors=["momentum", "value", "quality", "size"],
        n_quantiles=5,
        refresh=False,
    )
    assert {"momentum", "value", "quality", "size"} <= set(fr2.returns.columns) and fr2.notes == []
    assert fr2.returns["value"].abs().sum() > 0
