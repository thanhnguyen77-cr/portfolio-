from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from quant.technical import (
    adx,
    atr,
    bollinger,
    indicator_frame,
    macd,
    markov_regimes,
    mfi,
    nearest_levels,
    obv,
    relative_volume,
    rsi,
    rule_based_regime,
    stochastic,
    support_resistance,
    swing_points,
    trend_regime,
    volatility_regime,
    vwap,
)

WILDER = [
    44.34,
    44.09,
    44.15,
    43.61,
    44.33,
    44.83,
    45.10,
    45.42,
    45.84,
    46.08,
    45.89,
    46.03,
    45.61,
    46.28,
    46.28,
    46.00,
    46.03,
    46.41,
    46.22,
    45.64,
]


def test_rsi_matches_the_wilder_reference_and_extremes():
    r = rsi(pd.Series(WILDER), 14)
    assert np.isnan(r.iloc[13]) and r.iloc[14] == pytest.approx(70.46, abs=0.1)
    assert r.iloc[19] == pytest.approx(57.92, abs=0.6)
    assert rsi(pd.Series(np.arange(30, dtype=float)), 14).iloc[-1] == 100.0
    assert rsi(pd.Series(np.arange(30, 0, -1, dtype=float)), 14).iloc[-1] == 0.0
    with pytest.raises(ValueError):
        rsi(pd.Series(WILDER), 0)


def test_atr_constant_range():
    h, lo, c = pd.Series([11.0] * 30), pd.Series([9.0] * 30), pd.Series([10.0] * 30)
    a = atr(h, lo, c, 14)
    assert np.isnan(a.iloc[12]) and a.iloc[13] == 2.0 and a.iloc[-1] == 2.0


def _trending(n: int = 300, slope: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    close = pd.Series(100 + slope * np.arange(n) + rng.normal(0, 0.3, n))
    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    vol = pd.Series(rng.uniform(0.9e6, 1.1e6, n))
    return high, low, close, vol


def test_trend_and_momentum_indicators_on_an_uptrend():
    high, low, close, vol = _trending()
    m = macd(close)
    assert m["macd"].iloc[-1] > 0 and set(m.columns) == {"macd", "macd_signal", "macd_hist"}
    a = adx(high, low, close)
    assert a["adx"].iloc[-1] > 25 and a["plus_di"].iloc[-1] > a["minus_di"].iloc[-1]
    assert ((a["adx"].dropna() >= 0) & (a["adx"].dropna() <= 100)).all()
    st = stochastic(high, low, close)
    assert st["stoch_k"].dropna().between(0, 100).all() and st["stoch_k"].iloc[-1] > 60
    bb = bollinger(close).dropna()
    assert (bb["bb_upper"] > bb["bb_mid"]).all() and (bb["bb_mid"] > bb["bb_lower"]).all()
    assert bb["bb_pct_b"].iloc[-1] > 0.5
    assert rsi(close).iloc[-1] > 50


def test_volume_indicators():
    high, low, close, vol = _trending()
    assert obv(close, vol).iloc[-1] > 0
    f = mfi(high, low, close, vol)
    assert f.dropna().between(0, 100).all() and f.iloc[-1] > 50
    v = vwap(high, low, close, vol, window=20)
    ok = v.notna()
    assert (v[ok] >= low.rolling(20).min()[ok]).all() and (v[ok] <= high.rolling(20).max()[ok]).all()
    assert vwap(high, low, close, vol, window=None).notna().all()
    rv = relative_volume(pd.Series([1e6] * 60), 20)
    assert rv.iloc[-1] == pytest.approx(1.0)


def test_support_resistance_from_repeated_pivots():
    rng = np.random.default_rng(1)
    pattern = np.concatenate([np.linspace(100, 120, 20), np.linspace(120, 100, 20)] * 5)
    close = pd.Series(pattern + rng.normal(0, 0.2, len(pattern)))
    high, low = close + 0.3, close - 0.3
    piv = swing_points(high, low, left=5, right=5)
    assert piv["swing_high"].notna().sum() >= 4 and piv["swing_low"].notna().sum() >= 4
    levels = support_resistance(high, low, close, tolerance=0.02)
    assert len(levels) >= 2
    top = levels["level"].round(0).tolist()
    assert any(abs(lvl - 120) < 2 for lvl in top) and any(abs(lvl - 100) < 2 for lvl in top)
    assert (levels["touches"] >= 2).all()
    near = nearest_levels(levels, 110.0)
    assert near["support"] and near["resistance"] and near["support"][0] < 110 < near["resistance"][0]
    assert nearest_levels(levels.iloc[0:0], 110.0) == {"support": [], "resistance": []}


def test_volatility_and_trend_regimes():
    rng = np.random.default_rng(2)
    rets = np.concatenate([rng.normal(0, 0.004, 400), rng.normal(0, 0.03, 300)])
    close = pd.Series(100 * np.exp(np.cumsum(rets)))
    vr = volatility_regime(close)
    # the percentile is relative to the trailing year: right after the switch the new vol is extreme ...
    assert (vr["vol_regime"].iloc[420:520] == "high").mean() > 0.8
    # ... and once the whole trailing window is high-vol it reads as the new normal
    assert vr["vol_regime"].iloc[-1] in {"normal", "high"}
    assert vr["vol_percentile"].dropna().between(0, 1).all()
    up = pd.Series(100 + 0.3 * np.arange(400) + rng.normal(0, 0.5, 400))
    assert trend_regime(up)["trend_regime"].iloc[-1] == "uptrend"
    down = pd.Series(400 - 0.3 * np.arange(400) + rng.normal(0, 0.5, 400))
    assert trend_regime(down)["trend_regime"].iloc[-1] == "downtrend"
    high, low = up + 1, up - 1
    with_adx = trend_regime(up, high=high, low=low)
    assert "adx" in with_adx.columns and with_adx["trend_regime"].iloc[-1] == "uptrend"


def test_rule_based_regime_scores_and_labels():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2022-01-03", periods=400)
    rising = pd.Series(100 + 0.2 * np.arange(400) + rng.normal(0, 0.3, 400), index=idx)
    calm = rule_based_regime(rising, pd.Series(15.0, index=idx))
    assert calm["regime"].iloc[-1] == "risk_on" and calm["size_multiplier"].iloc[-1] == 1.0
    stressed = rule_based_regime(rising, pd.Series(35.0, index=idx))
    assert (
        stressed["regime"].iloc[-1] in {"caution", "risk_off"}
        and stressed["score"].iloc[-1] < calm["score"].iloc[-1]
    )
    falling = pd.Series(200 - 0.2 * np.arange(400) + rng.normal(0, 0.3, 400), index=idx)
    assert rule_based_regime(falling)["regime"].iloc[-1] == "risk_off"
    no_index = rule_based_regime(rising)
    assert no_index["regime"].iloc[-1] in {"risk_on", "caution"} and pd.isna(no_index["regime"].iloc[0])


def test_markov_regimes_recover_planted_volatility_segments():
    rng = np.random.default_rng(4)
    rets = np.concatenate(
        [rng.normal(0.0003, 0.005, 400), rng.normal(0.0, 0.03, 400), rng.normal(0.0003, 0.005, 400)]
    )
    r = pd.Series(rets, index=pd.bdate_range("2019-01-01", periods=len(rets)))
    mk = markov_regimes(r)
    assert mk.k_regimes == 2 and mk.variances[0] < mk.variances[1]
    assert np.allclose(mk.transition.sum(axis=1), 1.0, atol=1e-6)
    turbulent = mk.smoothed["regime_1"]
    assert (
        turbulent.iloc[400:800].mean() > 0.8
        and turbulent.iloc[:400].mean() < 0.2
        and turbulent.iloc[800:].mean() < 0.2
    )
    assert set(mk.regime.unique()) <= {0, 1} and mk.filtered.shape == (len(r), 2)


def test_indicator_frame(catalog):
    bars = make_bars(catalog["XNAS_AAPL"], date(2022, 1, 3), date(2024, 6, 28), seed=9).set_index(
        "session_date"
    )
    ind = indicator_frame(bars)
    expected = {
        "sma20",
        "sma50",
        "sma200",
        "ema20",
        "rsi14",
        "rsi2",
        "atr14",
        "atr14_pct",
        "macd",
        "adx",
        "bb_pct_b",
        "rv21",
        "relvol20",
        "high252",
        "dist_high252",
    }
    assert expected <= set(ind.columns) and ind.iloc[-1].notna().all()
