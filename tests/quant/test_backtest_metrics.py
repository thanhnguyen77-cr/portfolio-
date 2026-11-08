from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.metrics import benchmark_relative, compute_metrics, drawdown_series, max_drawdown


def test_constant_growth_metrics():
    idx = pd.bdate_range("2024-01-01", periods=253)
    eq = pd.Series(100.0 * 1.001 ** np.arange(253), index=idx)
    m = compute_metrics(eq)
    assert m["total_return"] == pytest.approx(1.001**252 - 1)
    years = (idx[-1] - idx[0]).days / 365.25
    assert m["cagr"] == pytest.approx((1.001**252) ** (1 / years) - 1)
    assert m["max_drawdown"] == 0.0 and m["longest_underwater_sessions"] == 0
    assert m["hit_rate"] == 1.0 and np.isnan(m["sharpe"])  # zero volatility: Sharpe undefined, not infinite
    assert set(m["returns_by_year"]) == {2024}


def test_drawdown_path():
    eq = pd.Series([100.0, 110.0, 99.0, 120.0, 108.0, 100.0], index=pd.bdate_range("2024-01-01", periods=6))
    mdd, longest = max_drawdown(eq)
    assert mdd == pytest.approx(100 / 120 - 1)
    assert longest == 2
    dd = drawdown_series(eq)
    assert dd.iloc[2] == pytest.approx(99 / 110 - 1) and dd.iloc[3] == 0.0


def test_sharpe_with_risk_free_and_benchmark_relative():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=504)
    rets = pd.Series(rng.normal(0.0005, 0.01, len(idx)), index=idx)
    eq = 100 * (1 + rets).cumprod()
    m0 = compute_metrics(eq)
    m_rf = compute_metrics(eq, rf=0.05)
    assert m_rf["sharpe"] < m0["sharpe"]
    assert m0["volatility"] == pytest.approx(rets.iloc[1:].std(ddof=1) * np.sqrt(252), rel=1e-6)
    rel = benchmark_relative(eq.pct_change().dropna(), eq, pd.Series(0.0, index=idx))
    assert rel["beta"] == pytest.approx(1.0) and abs(rel["alpha_annual"]) < 1e-9
    assert rel["tracking_error"] == pytest.approx(0.0, abs=1e-12) and rel["correlation"] == pytest.approx(1.0)
    full = compute_metrics(eq, benchmark=eq * 2)
    assert full["beta"] == pytest.approx(1.0)


def test_short_series():
    m = compute_metrics(pd.Series([100.0], index=pd.bdate_range("2024-01-01", periods=1)))
    assert m["sessions"] == 1 and m["total_return"] == 0.0
