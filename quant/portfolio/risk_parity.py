"""Risk budgeting: inverse volatility, equal risk contribution and arbitrary risk budgets.

Without box bounds the problem is solved in Spinu's (2013) convex form - minimise 0.5 y'Sigma y - sum(b_i
log y_i) and normalise - which is smooth and has a unique solution. With bounds it falls back to SLSQP on
the squared deviation of risk contributions from the budget, which is non-convex and reports success
honestly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.portfolio.covariance import align_weights
from quant.portfolio.mean_variance import Constraints, OptimisationError, OptimisationResult


def risk_contributions(weights: pd.Series | Mapping[str, float], cov: pd.DataFrame) -> pd.Series:
    """Fraction of portfolio variance attributable to each asset: w_i (Sigma w)_i / (w'Sigma w)."""
    w = align_weights(weights, cov)
    sigma = cov.to_numpy(dtype=float)
    marginal = sigma @ w
    total = float(w @ marginal)
    rc = w * marginal / total if total > 0 else np.full(len(w), np.nan)
    return pd.Series(rc, index=cov.index, name="risk_contribution")


def inverse_volatility(cov: pd.DataFrame) -> pd.Series:
    sd = np.sqrt(np.clip(np.diag(cov.to_numpy(dtype=float)), 1e-300, None))
    w = 1.0 / sd
    return pd.Series(w / w.sum(), index=cov.index, name="weight")


def _budget(cov: pd.DataFrame, budget: pd.Series | Mapping[str, float] | None) -> np.ndarray:
    n = cov.shape[0]
    if budget is None:
        return np.full(n, 1.0 / n)
    b = pd.Series(budget, dtype=float).reindex(cov.index)
    if b.isna().any() or (b <= 0).any():
        raise ValueError("risk budget must give a positive share to every asset in cov")
    return (b / b.sum()).to_numpy(dtype=float)


def risk_parity(
    cov: pd.DataFrame,
    budget: pd.Series | Mapping[str, float] | None = None,
    bounds: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
) -> OptimisationResult:
    """Weights whose risk contributions match `budget` (equal by default); long-only, fully invested."""
    assets = [str(a) for a in cov.index]
    sigma = np.atleast_2d(cov.to_numpy(dtype=float))
    b = _budget(cov, budget)
    sd = np.sqrt(np.clip(np.diag(sigma), 1e-300, None))
    x0 = (1.0 / sd) / np.sum(1.0 / sd)
    extras: dict[str, Any] = {"budget": pd.Series(b, index=cov.index, name="budget")}

    if bounds is None:

        def f(y: np.ndarray) -> float:
            return float(0.5 * y @ sigma @ y - b @ np.log(y))

        def g(y: np.ndarray) -> np.ndarray:
            return sigma @ y - b / y

        res = minimize(
            f,
            x0,
            jac=g,
            method="L-BFGS-B",
            bounds=[(1e-10, None)] * len(assets),
            options={"maxiter": 5000, "ftol": 1e-16, "gtol": 1e-12},
        )
        w = res.x / res.x.sum()
        method = "risk_parity"
    else:
        if isinstance(bounds, Mapping):
            box = [tuple(bounds.get(a, (0.0, 1.0))) for a in assets]
        else:
            box = [tuple(bounds)] * len(assets)

        def dev(w: np.ndarray) -> float:
            marginal = sigma @ w
            total = float(w @ marginal)
            if total <= 0:
                return 1e6
            rc = w * marginal / total
            return float(np.sum((rc - b) ** 2)) * 1e4

        res = minimize(
            dev,
            np.clip(x0, [lo for lo, _ in box], [hi for _, hi in box]),
            method="SLSQP",
            bounds=box,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
            options={"maxiter": 2000, "ftol": 1e-16},
        )
        w = res.x
        method = "risk_parity_bounded"
        extras["bounds"] = box
    if not np.all(np.isfinite(w)):
        raise OptimisationError(f"risk parity returned non-finite weights: {res.message}")
    weights = pd.Series(w, index=cov.index, name="weight")
    rc = risk_contributions(weights, cov)
    extras["risk_contributions"] = rc
    extras["max_budget_deviation"] = float(np.max(np.abs(rc.to_numpy() - b)))
    vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    cons = Constraints(long_only=True)
    return OptimisationResult(
        weights=weights,
        volatility=vol,
        expected_return=None,
        sharpe=None,
        method=method,
        success=bool(res.success),
        message=str(res.message),
        binding=cons.binding(weights) if bounds is None else [],
        extras=extras,
    )
