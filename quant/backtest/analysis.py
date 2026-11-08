"""Robustness analysis: the statistics that keep a backtest honest.

- walk-forward: choose parameters on a training window, evaluate on the following unseen window, repeat;
  the stitched out-of-sample curve is what a live deployment would have seen;
- parameter sensitivity: the same strategy on a grid of parameters; a result that only works for one cell
  is a curve-fit, not an edge;
- block bootstrap: a confidence interval for the Sharpe ratio that respects autocorrelation;
- deflated Sharpe ratio (Bailey & Lopez de Prado, 2014): the probability the observed Sharpe beats the
  Sharpe one would expect from the best of N noise trials, using the number of variants actually run and
  the non-normality of returns.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from quant.backtest.metrics import PERIODS_PER_YEAR, compute_metrics
from quant.backtest.result import BacktestResult
from quant.core.errors import BacktestError

Runner = Callable[[date, date, Mapping[str, Any]], BacktestResult]
EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------- walk-forward


@dataclass(frozen=True)
class WalkForwardWindow:
    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date


def walk_forward_windows(
    sessions: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    train_sessions: int,
    test_sessions: int,
    *,
    anchored: bool = False,
) -> list[WalkForwardWindow]:
    """Consecutive, non-overlapping test windows; training is the preceding `train_sessions` (or anchored)."""
    idx = pd.DatetimeIndex(sessions)
    if train_sessions < 2 or test_sessions < 1:
        raise BacktestError("train_sessions must be >= 2 and test_sessions >= 1")
    windows: list[WalkForwardWindow] = []
    start = train_sessions
    k = 0
    while start + test_sessions <= len(idx):
        tr0 = 0 if anchored else start - train_sessions
        windows.append(
            WalkForwardWindow(
                k,
                idx[tr0].date(),
                idx[start - 1].date(),
                idx[start].date(),
                idx[start + test_sessions - 1].date(),
            )
        )
        start += test_sessions
        k += 1
    if not windows:
        raise BacktestError("not enough sessions for a single walk-forward window")
    return windows


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame  # one row per window: dates, chosen params, train/test metric
    oos_equity: pd.Series  # stitched out-of-sample equity (chain-linked, starts at 1.0)
    oos_metrics: dict[str, Any]
    selection_metric: str
    results: list[BacktestResult] = field(default_factory=list)


def walk_forward(
    run: Runner,
    param_grid: Sequence[Mapping[str, Any]],
    sessions: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    train_sessions: int,
    test_sessions: int,
    *,
    select: str = "sharpe",
    anchored: bool = False,
    higher_is_better: bool = True,
) -> WalkForwardResult:
    """For each window pick the grid point with the best training `select` metric and run it out of sample."""
    if not param_grid:
        raise BacktestError("param_grid must contain at least one parameter set")
    windows = walk_forward_windows(sessions, train_sessions, test_sessions, anchored=anchored)
    rows: list[dict[str, Any]] = []
    oos_parts: list[pd.Series] = []
    results: list[BacktestResult] = []
    for w in windows:
        scored: list[tuple[float, Mapping[str, Any]]] = []
        for params in param_grid:
            res = run(w.train_start, w.train_end, params)
            value = res.metrics.get(select, float("nan"))
            scored.append((value if value is not None and math.isfinite(value) else -math.inf, params))
        best_value, best_params = max(scored, key=lambda t: t[0] if higher_is_better else -t[0])
        test = run(w.test_start, w.test_end, best_params)
        results.append(test)
        eq = test.equity_series
        rets = eq.pct_change().dropna()
        oos_parts.append(rets)
        rows.append(
            {
                "window": w.index,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "params": dict(best_params),
                f"train_{select}": best_value if math.isfinite(best_value) else float("nan"),
                f"test_{select}": test.metrics.get(select),
                "test_total_return": test.metrics.get("total_return"),
                "test_max_drawdown": test.metrics.get("max_drawdown"),
            }
        )
    oos_rets = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype="float64")
    oos_equity = (1.0 + oos_rets).cumprod()
    oos_metrics = compute_metrics(oos_equity) if len(oos_equity) > 1 else {}
    oos_metrics["windows"] = len(windows)
    return WalkForwardResult(pd.DataFrame(rows), oos_equity, oos_metrics, select, results)


# ----------------------------------------------------------------------------- sensitivity


def sensitivity_grid(
    run: Runner,
    base_params: Mapping[str, Any],
    grid: Mapping[str, Sequence[Any]],
    start: date,
    end: date,
    *,
    metrics: Sequence[str] = ("sharpe", "cagr", "max_drawdown", "total_return"),
) -> pd.DataFrame:
    """One run per combination of `grid` values layered over `base_params`: params + metrics per row."""
    keys = list(grid)
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = {**base_params, **dict(zip(keys, combo, strict=True))}
        res = run(start, end, params)
        row: dict[str, Any] = {k: params[k] for k in keys}
        for m in metrics:
            row[m] = res.metrics.get(m)
        row["fills"] = len(res.fills)
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_summary(
    table: pd.DataFrame, metric: str = "sharpe", threshold: float = 0.0
) -> dict[str, Any]:
    vals = pd.to_numeric(table[metric], errors="coerce").dropna()
    return {
        "metric": metric,
        "cells": int(len(table)),
        "fraction_above_threshold": float((vals > threshold).mean()) if len(vals) else float("nan"),
        "median": float(vals.median()) if len(vals) else float("nan"),
        "min": float(vals.min()) if len(vals) else float("nan"),
        "max": float(vals.max()) if len(vals) else float("nan"),
    }


# ----------------------------------------------------------------------------- bootstrap


def block_bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_boot: int = 1000,
    block: int = 20,
    ci: float = 0.95,
    seed: int = 0,
    rf: float = 0.0,
) -> dict[str, float]:
    """Circular block bootstrap of daily returns -> distribution of annualised Sharpe ratios."""
    r = returns.dropna().to_numpy(dtype="float64") - rf / PERIODS_PER_YEAR
    n = len(r)
    if n < max(block * 2, 20):
        raise BacktestError("too few returns for a block bootstrap")
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        s = x.std(ddof=1)
        return float(x.mean() / s * math.sqrt(PERIODS_PER_YEAR)) if s > 0 else float("nan")

    n_blocks = math.ceil(n / block)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        draws[b] = sharpe(r[idx[:n]])
    draws = draws[np.isfinite(draws)]
    alpha = (1 - ci) / 2
    return {
        "point": sharpe(r),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1 - alpha)),
        "prob_negative": float((draws < 0).mean()),
        "n_boot": int(len(draws)),
        "block": block,
        "ci": ci,
    }


# ----------------------------------------------------------------------------- deflated Sharpe


def expected_max_sharpe(n_trials: int, trial_var: float) -> float:
    """E[max SR] over `n_trials` noise trials with SR variance `trial_var` (per-period units)."""
    if n_trials <= 1 or trial_var <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(trial_var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurt: float) -> float:
    """P(true SR > sr_benchmark) from the per-period SR, sample size, skew and non-excess kurtosis."""
    if n_obs < 2:
        return float("nan")
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    *,
    trial_sharpes: Sequence[float] | None = None,
    rf: float = 0.0,
) -> dict[str, float]:
    """Deflated Sharpe ratio for a daily return series after `n_trials` variants were tried.

    `trial_sharpes` (annualised, one per variant run) gives the empirical SR variance across trials; without
    it the observed SR's own sampling variance is used as a conservative stand-in.
    """
    r = returns.dropna().to_numpy(dtype="float64") - rf / PERIODS_PER_YEAR
    n = len(r)
    if n < 10:
        raise BacktestError("too few returns for a deflated Sharpe ratio")
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else 0.0  # per period
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        trials_daily = np.asarray(trial_sharpes, dtype="float64") / math.sqrt(PERIODS_PER_YEAR)
        trial_var = float(trials_daily.var(ddof=1))
        n_trials = max(n_trials, len(trial_sharpes))
    else:
        trial_var = float((1 - skew * sr + (kurt - 1) / 4 * sr**2) / max(n - 1, 1))
    sr0 = expected_max_sharpe(n_trials, trial_var)
    dsr = probabilistic_sharpe(sr, sr0, n, skew, kurt)
    psr0 = probabilistic_sharpe(sr, 0.0, n, skew, kurt)
    return {
        "sharpe_annual": sr * math.sqrt(PERIODS_PER_YEAR),
        "sharpe_daily": sr,
        "n_trials": int(n_trials),
        "trial_var_daily": trial_var,
        "expected_max_sharpe_annual": sr0 * math.sqrt(PERIODS_PER_YEAR),
        "psr_vs_zero": psr0,
        "deflated_sharpe": dsr,
        "skew": skew,
        "kurtosis": kurt,
        "n_obs": int(n),
    }
