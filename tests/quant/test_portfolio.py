"""Portfolio construction: covariance estimators, optimisers, risk budgeting, BL, HRP, vol targeting,
sizing, and the vectorised rebalance pinned to the event-driven engine."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, zero_costs

from quant.backtest.engine import Backtest, BacktestConfig
from quant.portfolio import (
    Constraints,
    GroupLimit,
    View,
    apply_leverage,
    bl_weights,
    black_litterman,
    cov_to_corr,
    covariance,
    efficient_frontier,
    efficient_return,
    efficient_risk,
    ewma_cov,
    hrp,
    hrp_weights,
    implied_returns,
    inverse_volatility,
    kelly_fraction,
    ledoit_wolf_cov,
    max_sharpe,
    mean_variance,
    min_variance,
    nearest_psd,
    periodic_targets,
    portfolio_vol,
    realised_vol_leverage,
    returns_from_prices,
    risk_contributions,
    risk_parity,
    sample_cov,
    scale_to_target_vol,
    simulate_target_weights,
    size_position,
    wilson_lower_bound,
)
from quant.strategies import FixedWeights

ASSETS = ["A", "B", "C", "D"]


def _returns(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0003, 0.01, n)
    betas = np.array([1.0, 0.8, 1.2, 0.5])
    idio = rng.normal(0, [0.008, 0.012, 0.015, 0.006], (n, 4))
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame(f[:, None] * betas + idio, index=idx, columns=ASSETS)


def _cov(rho: float = 0.0, vols=(0.2, 0.3, 0.4)) -> pd.DataFrame:
    v = np.array(vols)
    corr = np.full((len(v), len(v)), rho)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.outer(v, v) * corr, index=ASSETS[: len(v)], columns=ASSETS[: len(v)])


# ---------------------------------------------------------------- covariance
def test_sample_cov_matches_numpy_and_is_psd():
    rets = _returns()
    est = sample_cov(rets)
    np.testing.assert_allclose(est.cov.to_numpy(), np.cov(rets.to_numpy().T, ddof=1) * 252, rtol=1e-10)
    assert np.linalg.eigvalsh(est.cov.to_numpy()).min() >= 0
    np.testing.assert_allclose(est.vol().to_numpy(), np.sqrt(np.diag(est.cov.to_numpy())))
    assert est.n_obs == 600 and est.method == "sample" and est.notes == []
    assert covariance(rets, "sample").cov.equals(est.cov)
    with pytest.raises(ValueError, match="unknown covariance method"):
        covariance(rets, "magic")


def test_ewma_limits_and_recency():
    rets = _returns()
    wide = ewma_cov(rets, halflife=1e9)
    np.testing.assert_allclose(wide.cov.to_numpy(), sample_cov(rets).cov.to_numpy(), rtol=1e-6)
    shocked = rets.copy()
    shocked.iloc[-60:, 0] *= 4.0  # a recent vol regime for A
    assert (
        ewma_cov(shocked, halflife=20).vol()["A"]
        > sample_cov(shocked).vol()["A"]
        > sample_cov(rets).vol()["A"]
    )
    assert any("effective sample size" in n for n in ewma_cov(rets, 20).notes)


def test_ledoit_wolf_shrinks_toward_identity_and_warns_on_short_samples():
    rets = _returns()
    lw, sc = ledoit_wolf_cov(rets), sample_cov(rets)
    assert 0 < lw.shrinkage < 1
    off = ~np.eye(4, dtype=bool)
    assert np.abs(lw.cov.to_numpy()[off]).sum() < np.abs(sc.cov.to_numpy()[off]).sum()
    assert np.linalg.eigvalsh(lw.cov.to_numpy()).min() >= 0
    short = ledoit_wolf_cov(rets.iloc[:6])
    assert any("poorly determined" in n for n in short.notes)


def test_nearest_psd_and_corr_helpers():
    bad = pd.DataFrame(
        [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]], index=ASSETS[:3], columns=ASSETS[:3]
    )
    assert np.linalg.eigvalsh(bad.to_numpy()).min() < 0
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() >= -1e-12
    np.testing.assert_allclose(fixed.to_numpy(), fixed.to_numpy().T)
    good = _cov(0.3)
    assert nearest_psd(good).equals(pd.DataFrame(good.to_numpy(), index=good.index, columns=good.columns))
    corr = cov_to_corr(good)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)
    assert corr.loc["A", "B"] == pytest.approx(0.3)
    assert portfolio_vol({"A": 1.0}, good) == pytest.approx(0.2)
    assert portfolio_vol({"A": 0.5, "B": 0.5}, _cov(0.0, (0.2, 0.2))) == pytest.approx(0.2 / np.sqrt(2))
    px = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    assert returns_from_prices(px)["A"].tolist() == pytest.approx([0.1, -0.1])
    assert returns_from_prices(px, "log")["A"].iloc[0] == pytest.approx(np.log(1.1))


# ---------------------------------------------------------------- mean-variance
def test_min_variance_two_uncorrelated_assets_is_analytic():
    cov = _cov(0.0, (0.2, 0.1))  # variances 0.04, 0.01
    res = min_variance(cov)
    assert res.success
    assert res.weights["A"] == pytest.approx(0.2, abs=1e-6) and res.weights["B"] == pytest.approx(
        0.8, abs=1e-6
    )
    assert res.volatility == pytest.approx(np.sqrt(0.2**2 * 0.04 + 0.8**2 * 0.01), abs=1e-6)
    assert res.expected_return is None and res.sharpe is None


def test_mean_variance_respects_box_and_group_limits_and_reports_binding():
    cov = _cov(0.2, (0.2, 0.25, 0.3, 0.35))
    mu = pd.Series([0.30, 0.12, 0.10, 0.08], index=ASSETS)
    cons = Constraints(upper=0.4, groups=[GroupLimit("tech", ("A", "B"), 0.0, 0.5)])
    res = mean_variance(mu, cov, risk_aversion=1.0, constraints=cons)
    assert res.success and res.weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert res.weights["A"] <= 0.4 + 1e-6 and res.weights["A"] + res.weights["B"] <= 0.5 + 1e-6
    assert "upper:A" in res.binding and "group:tech:upper" in res.binding
    free = mean_variance(mu, cov, risk_aversion=1.0)
    assert free.weights["A"] > 0.4  # the cap really bound
    assert free.expected_return > res.expected_return


def test_max_sharpe_recovers_tangency_portfolio():
    cov = _cov(0.2)
    mu = pd.Series([0.06, 0.09, 0.12], index=ASSETS[:3])
    rf = 0.02
    raw = np.linalg.solve(cov.to_numpy(), (mu - rf).to_numpy())
    tangency = raw / raw.sum()
    assert (tangency > 0).all()  # the unconstrained answer is feasible long-only
    res = max_sharpe(mu, cov, rf=rf)
    np.testing.assert_allclose(res.weights.to_numpy(), tangency, atol=1e-4)
    analytic_sr = float((mu - rf).to_numpy() @ tangency) / np.sqrt(tangency @ cov.to_numpy() @ tangency)
    assert res.sharpe == pytest.approx(analytic_sr, abs=1e-6)


def test_efficient_frontier_is_monotone_and_targets_behave():
    cov = _cov(0.2)
    mu = pd.Series([0.06, 0.09, 0.12], index=ASSETS[:3])
    fr = efficient_frontier(mu, cov, n_points=8)
    assert fr["success"].all() and len(fr) == 8
    assert (np.diff(fr["volatility"].to_numpy()) >= -1e-7).all()
    assert fr["volatility"].iloc[0] == pytest.approx(min_variance(cov).volatility, abs=1e-5)
    assert fr["expected_return"].iloc[-1] == pytest.approx(0.12, abs=1e-5)
    assert max_sharpe(mu, cov).sharpe >= fr["sharpe"].max() - 1e-6
    bad = efficient_return(mu, cov, target_return=0.2)
    assert not bad.success and "not attainable" in bad.message
    er = efficient_risk(mu, cov, target_vol=0.25)
    assert er.success and er.volatility <= 0.25 + 1e-6 and er.volatility == pytest.approx(0.25, abs=1e-4)
    too_low = efficient_risk(mu, cov, target_vol=0.01)
    assert not too_low.success


def test_long_short_dollar_neutral_constraints():
    cov = _cov(0.2, (0.2, 0.25, 0.3, 0.35))
    mu = pd.Series([0.10, 0.05, -0.02, -0.06], index=ASSETS)
    cons = Constraints(long_only=False, budget=0.0, max_gross=2.0)
    res = mean_variance(mu, cov, risk_aversion=0.5, constraints=cons)
    assert res.success
    assert res.weights.sum() == pytest.approx(0.0, abs=1e-6) and res.weights.abs().sum() <= 2.0 + 1e-6
    assert res.weights["A"] > 0 > res.weights["D"]
    assert "max_gross" in res.binding


# ---------------------------------------------------------------- risk budgeting
def test_risk_parity_equalises_contributions():
    cov = sample_cov(_returns()).cov
    res = risk_parity(cov)
    rc = risk_contributions(res.weights, cov)
    np.testing.assert_allclose(rc.to_numpy(), 0.25, atol=1e-6)
    assert res.weights.sum() == pytest.approx(1.0) and (res.weights > 0).all()
    assert res.extras["max_budget_deviation"] < 1e-6
    assert res.volatility == pytest.approx(portfolio_vol(res.weights, cov))


def test_risk_parity_budget_bounds_and_inverse_vol():
    cov = sample_cov(_returns()).cov
    budget = {"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.1}
    res = risk_parity(cov, budget=budget)
    np.testing.assert_allclose(res.extras["risk_contributions"].to_numpy(), [0.5, 0.2, 0.2, 0.1], atol=1e-6)
    bounded = risk_parity(cov, bounds=(0.05, 0.3))
    assert bounded.method == "risk_parity_bounded"
    assert (bounded.weights >= 0.05 - 1e-8).all() and (bounded.weights <= 0.3 + 1e-8).all()
    assert bounded.weights.sum() == pytest.approx(1.0, abs=1e-8)
    diag = _cov(0.0)
    np.testing.assert_allclose(
        risk_parity(diag).weights.to_numpy(), inverse_volatility(diag).to_numpy(), atol=1e-6
    )
    with pytest.raises(ValueError, match="positive share"):
        risk_parity(cov, budget={"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})


# ---------------------------------------------------------------- Black-Litterman
def test_black_litterman_no_views_is_equilibrium():
    cov = _cov(0.3)
    mkt = pd.Series([0.5, 0.3, 0.2], index=ASSETS[:3])
    pi = implied_returns(cov, mkt, risk_aversion=2.5)
    np.testing.assert_allclose(pi.to_numpy(), 2.5 * cov.to_numpy() @ mkt.to_numpy())
    bl = black_litterman(cov, mkt, views=(), risk_aversion=2.5, tau=0.05)
    pd.testing.assert_series_equal(bl.posterior_mean, bl.prior, check_names=False)
    np.testing.assert_allclose(bl_weights(bl.posterior_mean, cov, 2.5).to_numpy(), mkt.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(bl.posterior_cov.to_numpy(), cov.to_numpy() * 1.05)
    assert any("no views" in n for n in bl.notes)


def test_black_litterman_views_move_the_posterior():
    cov = _cov(0.3)
    mkt = pd.Series([0.5, 0.3, 0.2], index=ASSETS[:3])
    prior = implied_returns(cov, mkt)
    up = black_litterman(cov, mkt, [View({"A": 1.0}, prior["A"] + 0.05)])
    assert prior["A"] < up.posterior_mean["A"] < prior["A"] + 0.05
    certain = black_litterman(cov, mkt, [View({"A": 1.0}, prior["A"] + 0.05, confidence=1.0)])
    assert certain.posterior_mean["A"] == pytest.approx(prior["A"] + 0.05, abs=1e-6)
    rel = black_litterman(cov, mkt, [View({"A": 1.0, "B": -1.0}, (prior["A"] - prior["B"]) + 0.03)])
    assert (rel.posterior_mean["A"] - rel.posterior_mean["B"]) > (prior["A"] - prior["B"])
    assert rel.P.shape == (1, 3) and rel.omega.shape == (1, 1)
    with pytest.raises(ValueError, match="not in cov"):
        black_litterman(cov, mkt, [View({"Z": 1.0}, 0.1)])


# ---------------------------------------------------------------- HRP
def _block_cov(vol_cd: float = 0.2) -> pd.DataFrame:
    corr = np.array([[1, 0.9, 0, 0], [0.9, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]], dtype=float)
    v = np.array([0.2, 0.2, vol_cd, vol_cd])
    return pd.DataFrame(np.outer(v, v) * corr, index=ASSETS, columns=ASSETS)


def test_hrp_symmetric_blocks_and_variance_preference():
    w = hrp_weights(_block_cov())
    np.testing.assert_allclose(w.to_numpy(), 0.25, atol=1e-12)
    res = hrp(_block_cov(0.4))
    assert (res.weights >= 0).all() and res.weights.sum() == pytest.approx(1.0)
    assert res.weights["A"] + res.weights["B"] == pytest.approx(0.8, abs=1e-9)  # inverse cluster variance
    assert set(res.extras["order"]) == set(ASSETS) and res.method == "hrp"


def test_hrp_tolerates_singular_covariance():
    rets = _returns()
    rets["A2"] = rets["A"]  # perfectly collinear -> singular covariance
    cov = sample_cov(rets).cov
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(cov.to_numpy())
    w = hrp_weights(cov)
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and len(w) == 5


# ---------------------------------------------------------------- vol targeting
def test_scale_to_target_vol_and_leverage_cap():
    cov = _cov(0.2)
    w = pd.Series([0.4, 0.3, 0.3], index=ASSETS[:3])
    scaled, lev = scale_to_target_vol(w, cov, 0.10)
    assert portfolio_vol(scaled, cov) == pytest.approx(0.10, abs=1e-12)
    assert lev == pytest.approx(0.10 / portfolio_vol(w, cov))
    _, capped = scale_to_target_vol(w, cov, 5.0, max_leverage=1.5)
    assert capped == 1.5


def test_realised_vol_leverage_uses_only_past_information():
    r = _returns()["A"]
    lev = realised_vol_leverage(r, 0.10, window=21, max_leverage=3.0)
    poisoned = r.copy()
    poisoned.iloc[-1] *= 100
    pd.testing.assert_series_equal(realised_vol_leverage(poisoned, 0.10, window=21, max_leverage=3.0), lev)
    assert lev.iloc[:21].isna().all() and lev.iloc[21:].notna().all() and (lev.dropna() <= 3.0).all()
    levered = apply_leverage(r, lev, financing_rate=0.05).dropna()
    i = levered.index[0]
    assert levered[i] == pytest.approx(lev[i] * r[i] - (lev[i] - 1) * 0.05 / 252)
    with pytest.raises(ValueError, match="lag"):
        realised_vol_leverage(r, 0.1, lag=0)


# ---------------------------------------------------------------- sizing
def test_sizing_edge_regime_and_caps():
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.4) and kelly_fraction(0.5, 0.0) == -1.0
    assert wilson_lower_bound(0.6, 40) < 0.6 and wilson_lower_bound(0.6, 0) == 0.0
    base = dict(win_rate=0.6, payoff=2.0, entry=100.0, stop=95.0, capital=100_000.0)
    no_edge = size_position(win_rate=0.3, payoff=1.0, entry=100.0, stop=95.0, capital=100_000.0)
    assert no_edge.decision == "NO_TRADE" and no_edge.binding_constraint == "kelly"
    off = size_position(**base, regime="risk_off")
    assert off.decision == "NO_TRADE" and off.binding_constraint == "regime"
    res = size_position(**base)  # n_trades=0 -> no shrinkage; f=0.4, quarter Kelly -> 10% at risk
    assert res.kelly_full_raw == pytest.approx(0.4) and res.kelly_used == pytest.approx(0.1)
    assert res.caps_qty["kelly"] == 2000 and res.caps_qty["max_risk_per_trade"] == 200
    assert res.qty == 200 and res.binding_constraint == "max_risk_per_trade" and res.decision == "SIZE"
    assert res.amount_at_risk == pytest.approx(1000.0) and res.risk_pct_of_capital == pytest.approx(1.0)
    cashbound = size_position(**base, max_risk_per_trade_pct=100.0, cash=30_000.0, cash_reserve=5_000.0)
    assert cashbound.qty == 250 and cashbound.binding_constraint == "cash_reserve"
    shrunk = size_position(**base, n_trades=40)
    assert shrunk.inputs["win_rate_used"] < 0.6 and shrunk.kelly_full_conservative < 0.4
    lots = size_position(**base, lot_size=100, max_risk_per_trade_pct=1.5)  # 300 shares -> 300
    assert lots.qty % 100 == 0 and lots.qty == 300
    exposure = size_position(**base, max_total_exposure_pct=10.0, invested=9_000.0)
    assert exposure.qty == 10 and exposure.binding_constraint == "max_total_exposure"
    with pytest.raises(ValueError, match="stop must be below"):
        size_position(win_rate=0.6, payoff=2.0, entry=100.0, stop=101.0, capital=1.0)


# ---------------------------------------------------------------- rebalance == engine
def test_rebalance_matches_engine(catalog):
    start, end = date(2023, 1, 3), date(2024, 6, 28)
    ids = ["XNAS_AAPL", "XNAS_MSFT", "XNYS_JPM"]
    bars = pd.concat([make_bars(catalog[i], start, end, seed=k + 1) for k, i in enumerate(ids)])
    weights = {"XNAS_AAPL": 0.25, "XNAS_MSFT": 0.2, "XNYS_JPM": 0.15}  # gross 0.6: cash never binds
    closes = bars.pivot(index="session_date", columns="instrument_id", values="close").sort_index()
    opens = bars.pivot(index="session_date", columns="instrument_id", values="open").sort_index()
    targets = periodic_targets(closes.index, weights, "monthly")
    assert len(targets) == 18
    for fractional in (True, False):
        cfg = BacktestConfig(
            start=start,
            end=end,
            initial_cash=1_000_000.0,
            cost_model=zero_costs(),
            participation_cap=None,
            fractional_shares=fractional,
            min_notional=0.0,
        )
        res = Backtest(cfg, FixedWeights(ids, weights, "monthly"), bars, catalog).run()
        sim = simulate_target_weights(opens, closes, targets, 1_000_000.0, fractional=fractional)
        np.testing.assert_allclose(sim.equity.to_numpy(), res.equity_series.to_numpy(), rtol=1e-10)
        assert len(sim.trades) == len(res.fills) > 3 * 17
        engine_qty = res.positions.sort_values("session_date").groupby("instrument_id")["qty"].last()
        np.testing.assert_allclose(sim.holdings.iloc[-1].reindex(ids), engine_qty.reindex(ids))
        assert abs(sim.weights.iloc[-1].sum() - 0.6) < 0.1
    with pytest.raises(ValueError, match="not in the price index"):
        simulate_target_weights(opens, closes, targets.shift(1, freq="D"), 1.0)


def test_fixed_weights_never_rebalances_when_asked(catalog):
    start, end = date(2024, 1, 2), date(2024, 6, 28)
    ids = ["XNAS_AAPL", "XNAS_MSFT"]
    bars = pd.concat([make_bars(catalog[i], start, end, seed=k + 5) for k, i in enumerate(ids)])
    cfg = BacktestConfig(
        start=start, end=end, cost_model=zero_costs(), fractional_shares=True, min_notional=0.0
    )
    once = Backtest(cfg, FixedWeights(ids, rebalance="never"), bars, catalog).run()
    assert len(once.fills) == 2
    weekly = Backtest(cfg, FixedWeights(ids, rebalance="weekly"), bars, catalog).run()
    assert len(weekly.fills) > 40
    with pytest.raises(ValueError, match="rebalance must be"):
        FixedWeights(ids, rebalance="daily")
