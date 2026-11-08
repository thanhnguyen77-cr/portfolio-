"""Risk analytics: VaR family and backtest, stress windows and shocks, factor scenarios, correlation
structure, drawdown episodes, and the HTML risk report."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant.risk import (
    STRESS_WINDOWS,
    average_correlation,
    beta_shock,
    build_risk_report,
    calmar_ratio,
    conditional_drawdown_at_risk,
    cornish_fisher_var,
    cornish_fisher_z,
    correlation_breakdown,
    correlation_summary,
    diversification_ratio,
    drawdown_episodes,
    estimate_betas,
    factor_scenario,
    filtered_historical_var,
    historical_stress,
    historical_var,
    hypothetical_shocks,
    kupiec_test,
    monte_carlo_var,
    pain_index,
    parametric_var,
    portfolio_returns,
    rolling_absorption,
    rolling_historical_var,
    scenario_table,
    ulcer_index,
    underwater_summary,
    var_table,
)
from quant.risk.stress import StressWindow


def _normal_returns(n=2000, mu=0.0004, sd=0.012, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sd, n), index=pd.bdate_range("2015-01-01", periods=n), name="r")


# ---------------------------------------------------------------- VaR family
def test_historical_var_is_the_empirical_quantile_and_cvar_exceeds_var():
    r = _normal_returns()
    res = historical_var(r, 0.99)
    assert res.var == pytest.approx(-np.quantile(r, 0.01))
    assert res.cvar >= res.var and res.n_obs == len(r) and res.method == "historical"
    assert res.cvar == pytest.approx(-r[r <= -res.var].mean())
    multi = historical_var(r, 0.95, horizon=5)
    assert multi.var > historical_var(r, 0.95).var and "overlapping" in multi.notes[0]
    with pytest.raises(ValueError, match="alpha"):
        historical_var(r, 0.3)
    with pytest.raises(ValueError, match="at least"):
        historical_var(r.iloc[:10])


def test_parametric_var_matches_closed_form_and_t_is_fatter():
    r = _normal_returns()
    mu, sd = r.mean(), r.std(ddof=1)
    res = parametric_var(r, 0.99)
    assert res.var == pytest.approx(-mu + stats.norm.ppf(0.99) * sd)
    assert res.cvar == pytest.approx(-mu + sd * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01)
    assert res.cvar > res.var
    five = parametric_var(r, 0.99, horizon=5)
    assert five.var == pytest.approx(-5 * mu + stats.norm.ppf(0.99) * sd * np.sqrt(5))
    assert any("iid" in n for n in five.notes)
    rng = np.random.default_rng(1)
    fat = pd.Series(stats.t.rvs(4, size=4000, random_state=rng) * 0.01)
    t_res, n_res = parametric_var(fat, 0.99, dist="t"), parametric_var(fat, 0.99, dist="normal")
    assert t_res.var > n_res.var and 2 < t_res.params["df"] < 8
    assert t_res.cvar > t_res.var
    with pytest.raises(ValueError, match="dist"):
        parametric_var(r, dist="laplace")


def test_cornish_fisher_adjustment_direction():
    z = stats.norm.ppf(0.01)
    assert cornish_fisher_z(z, 0.0, 0.0) == pytest.approx(z)
    assert cornish_fisher_z(z, -1.0, 0.0) < z  # negative skew -> fatter left tail
    assert cornish_fisher_z(z, 0.0, 3.0) < z  # excess kurtosis -> fatter tails at 1%
    r = _normal_returns()
    cf, norm = cornish_fisher_var(r, 0.99), parametric_var(r, 0.99)
    assert cf.var == pytest.approx(norm.var, rel=0.1) and cf.cvar >= cf.var
    skewed = r.copy()
    skewed.iloc[::50] -= 0.08  # periodic crashes
    assert cornish_fisher_var(skewed, 0.99).var > parametric_var(skewed, 0.99).var


def test_filtered_historical_var_reacts_to_the_current_vol_regime():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.006, 700)
    wild = rng.normal(0, 0.02, 60)
    r = pd.Series(np.r_[calm, wild], index=pd.bdate_range("2020-01-01", periods=760))
    fhs = filtered_historical_var(r, 0.99)
    hist = historical_var(r, 0.99)
    assert np.isfinite(fhs.var) and fhs.cvar >= fhs.var
    assert fhs.var > hist.var  # the GARCH filter knows vol just jumped; the unconditional quantile does not
    assert fhs.params["sigma_next_daily"] > 0.01 and any("persistence" in n for n in fhs.notes)
    with pytest.raises(ValueError, match="at least 100"):
        filtered_historical_var(r.iloc[:80])


def test_monte_carlo_var_agrees_with_parametric_for_a_normal_portfolio():
    cov = pd.DataFrame([[0.0002, 0.00005], [0.00005, 0.0001]], index=["A", "B"], columns=["A", "B"])
    mu = pd.Series([0.0003, 0.0002], index=["A", "B"])
    w = pd.Series([0.6, 0.4], index=["A", "B"])
    mc = monte_carlo_var(w, mu, cov, 0.99, n_sims=400_000, seed=7)
    port_mu = float(w @ mu)
    port_sd = float(np.sqrt(w @ cov.to_numpy() @ w))
    assert mc.var == pytest.approx(-port_mu + stats.norm.ppf(0.99) * port_sd, rel=0.02)
    assert mc.cvar > mc.var
    t_mc = monte_carlo_var(w, mu, cov, 0.99, n_sims=200_000, dist="t", df=4, seed=7)
    assert t_mc.var > mc.var  # fat tails at equal variance
    with pytest.raises(ValueError, match="df"):
        monte_carlo_var(w, mu, cov, dist="t", df=2.0)


def test_var_table_reports_failures_instead_of_raising():
    r = _normal_returns(n=600)
    table = var_table(r, 0.99)
    assert set(table["method"]) == {
        "historical", "parametric_normal", "parametric_t", "cornish_fisher", "filtered_historical"
    }  # fmt: skip
    assert table["var"].notna().all() and (table["cvar"] >= table["var"] - 1e-12).all()
    short = var_table(r.iloc[:80], 0.99)
    row = short.set_index("method").loc["filtered_historical"]
    assert np.isnan(row["var"]) and row["notes"].startswith("failed")


def test_kupiec_backtest_accepts_calibrated_var_and_rejects_too_small_var():
    r = _normal_returns(n=3000, seed=11)
    rv = rolling_historical_var(r, 250, 0.99)
    assert rv.iloc[:250].isna().all()
    # no look-ahead: the VaR for day t is a quantile of returns strictly before t
    i = 300
    assert rv.iloc[i] == pytest.approx(-np.quantile(r.iloc[i - 250 : i], 0.01))
    ok = kupiec_test(r, rv, 0.99)
    assert ok["n"] == 3000 - 250 and ok["p_value"] > 0.05 and not ok["reject_at_5pct"]
    bad = kupiec_test(r, rv * 0.5, 0.99)
    assert bad["exceptions"] > 3 * ok["exceptions"] and bad["p_value"] < 0.01 and bad["reject_at_5pct"]
    none = kupiec_test(r, rv * 10, 0.99)
    assert none["exceptions"] == 0 and none["lr_pof"] > 0


# ---------------------------------------------------------------- stress and scenarios
def _long_prices() -> pd.DataFrame:
    idx = pd.bdate_range("2006-01-02", "2023-12-29")
    rng = np.random.default_rng(5)
    rets = pd.DataFrame(rng.normal(0.0002, 0.005, (len(idx), 2)), index=idx, columns=["A", "B"])
    covid = (idx >= pd.Timestamp("2020-02-20")) & (idx <= pd.Timestamp("2020-03-23"))
    rets.loc[covid, "A"] = -0.015  # a planted crash in A inside the COVID window
    return 100 * (1 + rets).cumprod()


def test_historical_stress_uses_planted_window_and_reports_coverage():
    px = _long_prices()
    w = {"A": 0.5, "B": 0.5}
    st = historical_stress(w, px).set_index("window")
    assert set(st.index) == set(STRESS_WINDOWS)
    assert st["usable"].all() and (st["coverage"] > 0.9).all()
    covid = st.loc["covid_2020"]
    n_days = int(covid["sessions"])
    expected_a = (1 - 0.015) ** n_days - 1
    assert covid["ret_A"] == pytest.approx(expected_a, rel=1e-9)
    assert covid["portfolio_return"] < -0.12  # ~ 0.5 * (-29%) + 0.5 * flat
    assert covid["max_drawdown"] <= covid["portfolio_return"] + 1e-12
    assert covid["worst_day"] <= -0.005
    short = historical_stress(w, px[px.index >= "2021-01-01"]).set_index("window")
    assert not short.loc["covid_2020", "usable"] and short.loc["covid_2020", "coverage"] == 0.0
    assert np.isnan(short.loc["covid_2020", "portfolio_return"]) and short.loc["gilt_crisis_2022", "usable"]
    custom = {"my_window": StressWindow("my_window", date(2010, 1, 4), date(2010, 1, 8), "test")}
    assert len(historical_stress(w, px, custom)) == 1


def test_hypothetical_and_beta_shocks_are_linear():
    w = pd.Series({"A": 0.6, "B": 0.4})
    table = hypothetical_shocks(w, {"equity_down": {"A": -0.2, "B": -0.1}, "only_a": {"A": -0.5, "Z": 1.0}})
    row = table.set_index("scenario")
    assert row.loc["equity_down", "portfolio_return"] == pytest.approx(-0.16)
    assert (
        row.loc["only_a", "portfolio_return"] == pytest.approx(-0.30)
        and row.loc["only_a", "ignored_assets"] == "Z"
    )
    assert beta_shock(w, {"A": 1.2, "B": 0.8}, -0.1) == pytest.approx(-(0.6 * 1.2 + 0.4 * 0.8) * 0.1)
    with pytest.raises(ValueError, match="betas missing"):
        beta_shock(w, {"A": 1.0}, -0.1)


def test_factor_scenarios_recover_planted_betas():
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2020-01-01", periods=750)
    factors = pd.DataFrame({"mkt": rng.normal(0, 0.01, 750), "rates": rng.normal(0, 0.004, 750)}, index=idx)
    true = pd.DataFrame({"mkt": [1.2, 0.6], "rates": [-0.5, 2.0]}, index=["A", "B"])
    noise = rng.normal(0, 0.002, (750, 2))
    rets = pd.DataFrame(factors.to_numpy() @ true.to_numpy().T + noise, index=idx, columns=["A", "B"])
    betas = estimate_betas(rets, factors)
    np.testing.assert_allclose(betas[["mkt", "rates"]].to_numpy(), true.to_numpy(), atol=0.05)
    assert (betas["r2"] > 0.9).all()
    w = {"A": 0.5, "B": 0.5}
    res = factor_scenario(w, betas, {"mkt": -0.2, "rates": 0.05})
    expected = sum(w[a] * (betas.loc[a, "mkt"] * -0.2 + betas.loc[a, "rates"] * 0.05) for a in w)
    assert res["portfolio_return"] == pytest.approx(expected)
    assert res["by_asset"].sum() == pytest.approx(expected) and res["by_factor"].sum() == pytest.approx(
        expected
    )
    table = scenario_table(w, betas, {"crash": {"mkt": -0.3}, "rates_up": {"rates": 0.1}})
    assert list(table["scenario"]) == ["crash", "rates_up"] and table["portfolio_return"].iloc[0] < 0
    with pytest.raises(ValueError, match="factors not in betas"):
        factor_scenario(w, betas, {"fx": 0.1})
    with pytest.raises(ValueError, match="no betas"):
        factor_scenario({"A": 1.0, "C": 0.0}, betas, {"mkt": 0.1})


# ---------------------------------------------------------------- correlation
def test_correlation_breakdown_and_diversification():
    rng = np.random.default_rng(2)
    n = 2000
    common = rng.normal(0, 0.01, n)
    stress = rng.random(n) < 0.1
    load = np.where(stress, 1.0, 0.2)  # assets co-move much more in stress days
    rets = pd.DataFrame(
        {k: load * common + rng.normal(0, 0.008, n) for k in ["A", "B", "C"]},
        index=pd.bdate_range("2015-01-01", periods=n),
    )
    marker = pd.Series(np.where(stress, -1.0, 0.0) + rng.normal(0, 1e-3, n), index=rets.index)
    br = correlation_breakdown(rets, reference=marker)  # bottom decile of the marker == the stress days
    assert br["avg_stress"] > br["avg_calm"] + 0.2 and br["n_stress"] + br["n_calm"] == n
    assert "external reference" in br["note"]
    by_mean = correlation_breakdown(rets)  # default reference: conditioning on own returns is biased
    assert "biased" in by_mean["note"] and by_mean["n_stress"] == br["n_stress"]
    assert (br["difference"].to_numpy()[~np.eye(3, dtype=bool)] > 0).all()
    assert average_correlation(pd.DataFrame(np.eye(100), columns=list("ABCDEFGHIJ")[:0] or range(100))) < 0.05
    roll = average_correlation(rets, window=63)
    assert isinstance(roll, pd.Series) and roll.iloc[:62].isna().all() and roll.dropna().between(-1, 1).all()
    cov = pd.DataFrame(np.eye(4) * 0.04, index=list("ABCD"), columns=list("ABCD"))
    assert diversification_ratio({c: 0.25 for c in "ABCD"}, cov) == pytest.approx(2.0)
    assert diversification_ratio({"A": 1.0}, cov) == pytest.approx(1.0)
    ar = rolling_absorption(rets, window=250, n_eigen=1)
    assert ar.dropna().between(0, 1).all()
    summary = correlation_summary(rets, cov=rets.cov() * 252, weights={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
    assert {"avg_correlation", "avg_correlation_stress", "absorption_ratio", "diversification_ratio"} <= set(
        summary
    )


# ---------------------------------------------------------------- drawdowns
def _equity() -> pd.Series:
    path = [100, 110, 120, 108, 96, 100, 125, 130, 117, 120, 135, 128]
    return pd.Series(path, index=pd.bdate_range("2024-01-01", periods=len(path)), dtype=float)


def test_drawdown_episodes_and_summaries():
    eq = _equity()
    table = drawdown_episodes(eq)
    assert len(table) == 3
    worst = table.iloc[0]
    assert (
        worst["peak"] == eq.index[2] and worst["trough"] == eq.index[4] and worst["recovery"] == eq.index[6]
    )
    assert worst["depth"] == pytest.approx(96 / 120 - 1)
    assert (
        worst["to_trough_sessions"] == 2 and worst["recovery_sessions"] == 2 and worst["total_sessions"] == 4
    )
    open_dd = table[table["recovery"].isna()]
    assert (
        len(open_dd) == 1
        and open_dd.iloc[0]["peak"] == eq.index[10]
        and np.isnan(open_dd.iloc[0]["recovery_sessions"])
    )
    s = underwater_summary(eq)
    assert s["max_drawdown"] == pytest.approx(-0.2) and s["current_drawdown"] == pytest.approx(128 / 135 - 1)
    assert 0 < s["time_underwater"] < 1 and s["ulcer_index"] == pytest.approx(ulcer_index(eq))
    assert pain_index(eq) > 0 and conditional_drawdown_at_risk(eq, 0.9) >= pain_index(eq)
    assert conditional_drawdown_at_risk(eq, 0.9) <= 0.2 + 1e-12
    assert calmar_ratio(eq) > 0
    assert drawdown_episodes(pd.Series([1.0, 2.0, 3.0])).empty


# ---------------------------------------------------------------- report
def test_risk_report_renders_with_caveats(tmp_path):
    px = _long_prices()
    rets = px.pct_change().iloc[1:]
    w = {"A": 0.5, "B": 0.5}
    port = portfolio_returns(rets, w)
    assert port.iloc[0] == pytest.approx(0.5 * rets.iloc[0].sum())
    out = tmp_path / "risk.html"
    html = build_risk_report(rets, w, prices=px, cov=rets.cov() * 252, alpha=0.99, path=out)
    assert out.exists() and "Caveats" in html and "Value at Risk" in html and "Kupiec" in html
    assert "Historical stress windows" in html and "covid_2020" in html and "Drawdowns" in html
    assert "describe that history" in html
    short = build_risk_report(rets.iloc[-200:], w)
    assert "too short" in short and "Historical stress windows" not in short
