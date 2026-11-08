from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.stats import (
    absorption_ratio,
    adf_test,
    engle_granger,
    fit_garch,
    forecast_vol,
    half_life,
    johansen,
    kpss_test,
    ols,
    pca,
    phillips_perron,
    spread_zscore,
    stationarity_summary,
    variance_ratio,
)


def _ar1(n: int, phi: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    eps = rng.normal(0, sd, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def test_unit_root_tests_agree_on_noise_and_random_walks():
    rng = np.random.default_rng(10)
    noise = pd.Series(rng.normal(0, 1, 600))
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 600)))
    assert adf_test(noise).reject_at_5pct and phillips_perron(noise).reject_at_5pct
    kp = kpss_test(noise)
    assert kp.statistic < kp.critical_values["1%"]  # KPSS has size distortions at 5%; must not reject at 1%
    assert not adf_test(walk).reject_at_5pct and kpss_test(walk).reject_at_5pct
    assert stationarity_summary(noise)["verdict"] in {"stationary", "inconclusive"}
    assert stationarity_summary(walk)["verdict"] == "unit_root"
    assert adf_test(noise).null == "unit root" and kpss_test(noise).null == "stationary"
    assert variance_ratio(pd.Series(_ar1(1000, 0.5, 1.0, rng))).reject_at_5pct
    with pytest.raises(ValueError):
        adf_test(pd.Series([1.0, 2.0, 3.0]))


def test_engle_granger_and_johansen_detect_a_planted_pair():
    rng = np.random.default_rng(11)
    x = pd.Series(100 + np.cumsum(rng.normal(0, 1, 800)))
    y = 2.0 * x + 1.0 + pd.Series(_ar1(800, 0.5, 0.5, rng))
    eg = engle_granger(y, x)
    assert eg.cointegrated_at_5pct and eg.pvalue < 0.01
    assert eg.hedge_ratio == pytest.approx(2.0, abs=0.05) and eg.intercept == pytest.approx(1.0, abs=1.0)
    assert 0 < eg.half_life_sessions < 5
    jo = johansen(pd.concat([y.rename("y"), x.rename("x")], axis=1))
    assert jo.rank_at_95 >= 1
    assert jo.cointegrating_vector["y"] == 1.0 and jo.cointegrating_vector["x"] == pytest.approx(
        -2.0, abs=0.3
    )
    other = pd.Series(50 + np.cumsum(rng.normal(0, 1, 800)))
    assert engle_granger(other, x).pvalue > 0.05
    assert johansen(pd.concat([x.rename("x"), other.rename("o")], axis=1)).rank_at_95 == 0
    with pytest.raises(ValueError):
        engle_granger(y.iloc[:10], x.iloc[:10])


def test_half_life_and_zscore():
    rng = np.random.default_rng(12)
    s = pd.Series(_ar1(3000, 0.9, 1.0, rng))
    assert half_life(s) == pytest.approx(-np.log(2) / np.log(0.9), rel=0.25)
    assert half_life(pd.Series(np.cumsum(rng.normal(0, 1, 500)))) > 20  # no mean reversion: long or infinite
    z = spread_zscore(s, 60)
    assert np.isnan(z.iloc[58]) and z.iloc[60:].abs().max() < 10
    full = spread_zscore(s)
    assert full.mean() == pytest.approx(0.0, abs=1e-9) and full.std(ddof=1) == pytest.approx(1.0)


def test_garch_recovers_planted_parameters_and_forecasts():
    rng = np.random.default_rng(13)
    n, omega, alpha, beta = 3000, 0.05, 0.10, 0.80
    var = np.empty(n)
    r = np.empty(n)
    var[0] = omega / (1 - alpha - beta)
    r[0] = rng.normal(0, np.sqrt(var[0]))
    for t in range(1, n):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
        r[t] = rng.normal(0, np.sqrt(var[t]))
    rets = pd.Series(r / 100.0, index=pd.bdate_range("2012-01-02", periods=n))  # back to decimal
    res = fit_garch(rets, dist="normal")
    assert res.converged and res.model == "GARCH"
    assert res.params["alpha[1]"] == pytest.approx(alpha, abs=0.05) and res.params[
        "beta[1]"
    ] == pytest.approx(beta, abs=0.08)
    assert 0.8 < res.persistence < 1.0
    assert res.conditional_vol.gt(0).all() and 0.05 < res.conditional_vol.mean() < 0.25
    assert res.std_resid.var() == pytest.approx(1.0, abs=0.15)
    fc = forecast_vol(res, 10)
    assert len(fc) == 10 and fc.gt(0).all()
    assert fit_garch(rets, model="GJR", dist="t").params.get("gamma[1]") is not None
    assert fit_garch(rets.iloc[:800], model="EGARCH", dist="normal").conditional_vol.gt(0).all()
    with pytest.raises(ValueError):
        fit_garch(rets.iloc[:50])


def test_pca_and_absorption_ratio():
    rng = np.random.default_rng(14)
    common = rng.normal(0, 0.01, 500)
    panel = pd.DataFrame(
        {f"a{i}": 0.8 * common + rng.normal(0, 0.005, 500) for i in range(3)}
        | {"b": rng.normal(0, 0.01, 500)},
        index=pd.bdate_range("2022-01-03", periods=500),
    )
    res = pca(panel)
    assert res.explained_variance_ratio.sum() == pytest.approx(1.0)
    assert (
        res.explained_variance_ratio["pc1"] > 0.5
        and res.loadings.shape == (4, 4)
        and res.scores.shape == (500, 4)
    )
    assert pca(panel, n_components=2).loadings.shape == (4, 2)
    ar = absorption_ratio(panel, n_eigen=1)
    assert 0.5 < ar <= 1.0
    rolling = absorption_ratio(panel, n_eigen=1, window=120)
    assert rolling.iloc[:119].isna().all() and rolling.dropna().between(0, 1).all()
    with pytest.raises(ValueError):
        pca(panel.iloc[:3])


def test_ols_recovers_coefficients_with_diagnostics():
    rng = np.random.default_rng(15)
    n = 800
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"x1": x1, "x2": x2})  # noqa: N806
    res = ols(pd.Series(y), X)
    assert res.params["const"] == pytest.approx(1.0, abs=0.1) and res.params["x1"] == pytest.approx(
        2.0, abs=0.1
    )
    assert res.params["x2"] == pytest.approx(-0.5, abs=0.1) and res.r2 > 0.9 and res.nobs == n
    d = res.diagnostics
    assert 1.7 < d["durbin_watson"] < 2.3 and d["breusch_pagan_p"] > 0.05 and d["breusch_godfrey_p"] > 0.01
    assert (res.vif < 1.5).all() and set(res.vif.index) == {"x1", "x2"} and res.cov_type == "nonrobust"
    assert set(res.table().columns) == {"coef", "se", "t", "p"}
    hetero = 1.0 + 2.0 * x1 + rng.normal(0, 1, n) * np.exp(0.5 * x1)  # variance rises with x1
    hd = ols(pd.Series(hetero), pd.Series(x1, name="x1")).diagnostics
    assert hd["breusch_pagan_p"] < 0.05 and hd["white_p"] < 0.05
    hac = ols(pd.Series(y), X, hac_lags=5)
    assert hac.cov_type == "HAC(5)" and np.isfinite(hac.bse).all() and hac.vif.index.tolist() == ["x1", "x2"]
    single = ols(pd.Series(y), pd.Series(x1, name="x1"))
    assert single.vif.empty and "cusum_p" in single.diagnostics
