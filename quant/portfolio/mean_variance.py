"""Mean-variance optimisation (SLSQP): minimum variance, risk-aversion utility, target return, target risk,
maximum Sharpe and the efficient frontier under box, budget, gross-exposure and group constraints.

Expected returns are an *input*; the optimiser magnifies whatever estimation error they carry (the classic
mean-variance failure mode), which is why every result reports the constraints that bind. Inputs are
annualised; volatility and Sharpe in the results are on the same basis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant.core.errors import QuantError


class OptimisationError(QuantError):
    """The optimiser could not produce a usable weight vector."""


@dataclass(frozen=True)
class GroupLimit:
    name: str
    members: tuple[str, ...]
    lower: float = 0.0
    upper: float = 1.0


@dataclass
class Constraints:
    long_only: bool = True
    lower: float | Mapping[str, float] | None = None  # default 0 when long_only, else -1
    upper: float | Mapping[str, float] = 1.0
    budget: float | None = 1.0  # sum of weights; None leaves it free
    max_gross: float | None = None  # sum of |weights|
    groups: Sequence[GroupLimit] = ()

    def bounds(self, assets: Sequence[str]) -> list[tuple[float, float]]:
        default_lo = 0.0 if self.long_only else -1.0
        out: list[tuple[float, float]] = []
        for a in assets:
            if isinstance(self.lower, Mapping):
                lo = float(self.lower.get(a, default_lo))
            else:
                lo = default_lo if self.lower is None else float(self.lower)
            hi = float(self.upper.get(a, 1.0)) if isinstance(self.upper, Mapping) else float(self.upper)
            if self.long_only:
                lo = max(lo, 0.0)
            if lo > hi:
                raise ValueError(f"lower bound {lo} exceeds upper bound {hi} for {a}")
            out.append((lo, hi))
        return out

    def scipy_constraints(self, assets: Sequence[str]) -> list[dict[str, Any]]:
        cons: list[dict[str, Any]] = []
        pos = {a: i for i, a in enumerate(assets)}
        if self.budget is not None:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda w, b=self.budget: float(np.sum(w) - b),
                    "jac": lambda w: np.ones_like(w),
                }
            )
        if self.max_gross is not None:
            cons.append({"type": "ineq", "fun": lambda w, g=self.max_gross: float(g - np.sum(np.abs(w)))})
        for g in self.groups:
            sel = np.array([pos[m] for m in g.members if m in pos], dtype=int)
            if len(sel) == 0:
                continue
            cons.append({"type": "ineq", "fun": lambda w, s=sel, lo=g.lower: float(np.sum(w[s]) - lo)})
            cons.append({"type": "ineq", "fun": lambda w, s=sel, hi=g.upper: float(hi - np.sum(w[s]))})
        return cons

    def binding(self, weights: pd.Series, tol: float = 1e-6) -> list[str]:
        """Human-readable list of constraints at their limit (bounds, groups, gross)."""
        assets = list(weights.index)
        out: list[str] = []
        for a, (lo, hi) in zip(assets, self.bounds(assets), strict=True):
            w = float(weights[a])
            if abs(w - lo) <= tol and not (self.long_only and lo == 0.0 and abs(w) <= tol):
                out.append(f"lower:{a}")
            if abs(w - hi) <= tol:
                out.append(f"upper:{a}")
        for g in self.groups:
            total = float(weights.reindex(list(g.members)).fillna(0.0).sum())
            if abs(total - g.lower) <= tol:
                out.append(f"group:{g.name}:lower")
            if abs(total - g.upper) <= tol:
                out.append(f"group:{g.name}:upper")
        if self.max_gross is not None and abs(float(weights.abs().sum()) - self.max_gross) <= tol:
            out.append("max_gross")
        return out


@dataclass
class OptimisationResult:
    weights: pd.Series
    volatility: float
    expected_return: float | None
    sharpe: float | None
    method: str
    success: bool
    message: str
    binding: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def risk_contributions(self, cov: pd.DataFrame) -> pd.Series:
        from quant.portfolio.risk_parity import risk_contributions

        return risk_contributions(self.weights, cov)

    def summary(self, cov: pd.DataFrame | None = None) -> pd.DataFrame:
        out = pd.DataFrame({"weight": self.weights})
        if cov is not None:
            out["risk_contribution"] = self.risk_contributions(cov)
        return out


# ---------------------------------------------------------------------- helpers
def _prepare(
    mu: pd.Series | Mapping[str, float] | None, cov: pd.DataFrame
) -> tuple[list[str], np.ndarray | None, np.ndarray]:
    assets = [str(a) for a in cov.index]
    sigma = np.atleast_2d(cov.to_numpy(dtype=float))
    if mu is None:
        return assets, None, sigma
    m = pd.Series(mu, dtype=float).reindex(cov.index)
    if m.isna().any():
        raise ValueError(f"expected returns missing for {list(m[m.isna()].index)}")
    return assets, m.to_numpy(dtype=float), sigma


def _start(assets: Sequence[str], cons: Constraints) -> np.ndarray:
    bounds = cons.bounds(assets)
    target = cons.budget if cons.budget is not None else 1.0
    x0 = np.full(len(assets), target / len(assets))
    return np.array([min(max(x, lo), hi) for x, (lo, hi) in zip(x0, bounds, strict=True)])


def _finish(
    x: np.ndarray,
    res: Any,
    assets: Sequence[str],
    mu: np.ndarray | None,
    sigma: np.ndarray,
    cons: Constraints,
    method: str,
    rf: float = 0.0,
    **extras: Any,
) -> OptimisationResult:
    w = np.where(np.abs(x) < 1e-12, 0.0, x)
    weights = pd.Series(w, index=list(assets), name="weight")
    vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    er = float(mu @ w) if mu is not None else None
    sharpe = (er - rf * float(np.sum(w))) / vol if (er is not None and vol > 0) else None
    return OptimisationResult(
        weights=weights,
        volatility=vol,
        expected_return=er,
        sharpe=sharpe,
        method=method,
        success=bool(res.success),
        message=str(res.message),
        binding=cons.binding(weights),
        extras={"iterations": int(getattr(res, "nit", 0)), **extras},
    )


def _solve(fun, jac, x0, bounds, scipy_cons):  # noqa: ANN001,ANN202
    res = minimize(
        fun,
        x0,
        jac=jac,
        bounds=bounds,
        constraints=scipy_cons,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not np.all(np.isfinite(res.x)):
        raise OptimisationError(f"optimiser returned non-finite weights: {res.message}")
    return res


# ---------------------------------------------------------------------- optimisers
def min_variance(cov: pd.DataFrame, constraints: Constraints | None = None) -> OptimisationResult:
    cons = constraints or Constraints()
    assets, _, sigma = _prepare(None, cov)
    res = _solve(
        lambda w: float(w @ sigma @ w),
        lambda w: 2.0 * sigma @ w,
        _start(assets, cons),
        cons.bounds(assets),
        cons.scipy_constraints(assets),
    )
    return _finish(res.x, res, assets, None, sigma, cons, "min_variance")


def mean_variance(
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    constraints: Constraints | None = None,
    rf: float = 0.0,
) -> OptimisationResult:
    """Maximise mu'w - (risk_aversion / 2) w'Sigma w."""
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    cons = constraints or Constraints()
    assets, m, sigma = _prepare(mu, cov)
    assert m is not None
    lam = float(risk_aversion)
    res = _solve(
        lambda w: float(-m @ w + 0.5 * lam * (w @ sigma @ w)),
        lambda w: -m + lam * sigma @ w,
        _start(assets, cons),
        cons.bounds(assets),
        cons.scipy_constraints(assets),
    )
    return _finish(res.x, res, assets, m, sigma, cons, "mean_variance", rf, risk_aversion=lam)


def efficient_return(
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    target_return: float,
    constraints: Constraints | None = None,
    rf: float = 0.0,
) -> OptimisationResult:
    """Minimum variance subject to expected return >= target_return."""
    cons = constraints or Constraints()
    assets, m, sigma = _prepare(mu, cov)
    assert m is not None
    extra = [{"type": "ineq", "fun": lambda w: float(m @ w - target_return), "jac": lambda w: m}]
    res = _solve(
        lambda w: float(w @ sigma @ w),
        lambda w: 2.0 * sigma @ w,
        _start(assets, cons),
        cons.bounds(assets),
        cons.scipy_constraints(assets) + extra,
    )
    out = _finish(res.x, res, assets, m, sigma, cons, "efficient_return", rf, target_return=target_return)
    if out.expected_return is not None and out.expected_return < target_return - 1e-6:
        out.success = False
        out.message = f"target return {target_return:.4f} is not attainable under the constraints"
    return out


def efficient_risk(
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    target_vol: float,
    constraints: Constraints | None = None,
    rf: float = 0.0,
) -> OptimisationResult:
    """Maximum expected return subject to volatility <= target_vol."""
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    cons = constraints or Constraints()
    assets, m, sigma = _prepare(mu, cov)
    assert m is not None
    extra = [
        {
            "type": "ineq",
            "fun": lambda w: float(target_vol**2 - w @ sigma @ w),
            "jac": lambda w: -2.0 * sigma @ w,
        }
    ]
    x0 = min_variance(cov, cons).weights.to_numpy()
    res = _solve(
        lambda w: float(-m @ w),
        lambda w: -m,
        x0,
        cons.bounds(assets),
        cons.scipy_constraints(assets) + extra,
    )
    out = _finish(res.x, res, assets, m, sigma, cons, "efficient_risk", rf, target_vol=target_vol)
    if out.volatility > target_vol + 1e-6:
        out.success = False
        out.message = f"minimum attainable volatility {out.volatility:.4f} exceeds target {target_vol:.4f}"
    return out


def max_sharpe(
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    rf: float = 0.0,
    constraints: Constraints | None = None,
) -> OptimisationResult:
    """Maximise (mu'w - rf * sum(w)) / sqrt(w'Sigma w) directly (non-convex; several starts, best kept)."""
    cons = constraints or Constraints()
    assets, m, sigma = _prepare(mu, cov)
    assert m is not None
    excess = m - rf

    def neg_sharpe(w: np.ndarray) -> float:
        var = float(w @ sigma @ w)
        if var <= 1e-18:
            return 1e6
        return float(-(excess @ w) / np.sqrt(var))

    def grad(w: np.ndarray) -> np.ndarray:
        var = float(w @ sigma @ w)
        if var <= 1e-18:
            return np.zeros_like(w)
        sd = np.sqrt(var)
        e = float(excess @ w)
        return -(excess / sd - e * (sigma @ w) / sd**3)

    starts = [_start(assets, cons), min_variance(cov, cons).weights.to_numpy()]
    best = None
    for x0 in starts:
        res = _solve(neg_sharpe, grad, x0, cons.bounds(assets), cons.scipy_constraints(assets))
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return _finish(best.x, best, assets, m, sigma, cons, "max_sharpe", rf)


def efficient_frontier(
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    n_points: int = 20,
    constraints: Constraints | None = None,
    rf: float = 0.0,
) -> pd.DataFrame:
    """Frontier from the minimum-variance return to the maximum attainable return, `n_points` rows."""
    cons = constraints or Constraints()
    assets, m, sigma = _prepare(mu, cov)
    assert m is not None
    low = min_variance(cov, cons)
    hi_res = _solve(
        lambda w: float(-m @ w),
        lambda w: -m,
        _start(assets, cons),
        cons.bounds(assets),
        cons.scipy_constraints(assets),
    )
    r_lo = float(m @ low.weights.to_numpy())
    r_hi = float(m @ hi_res.x)
    rows = []
    for target in np.linspace(r_lo, r_hi, max(2, n_points)):
        r = efficient_return(mu, cov, float(target), cons, rf)
        row = {
            "target_return": float(target),
            "expected_return": r.expected_return,
            "volatility": r.volatility,
            "sharpe": r.sharpe,
            "success": r.success,
        }
        row.update({f"w_{a}": float(r.weights[a]) for a in assets})
        rows.append(row)
    return pd.DataFrame(rows)
