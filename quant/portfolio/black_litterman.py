"""Black-Litterman: equilibrium (reverse-optimised) returns blended with investor views.

Prior: pi = delta * Sigma * w_mkt (excess returns implied by market-cap weights and a risk aversion delta).
Views: P (k x n pick matrix), Q (k view returns), Omega (k x k view uncertainty). Posterior mean
    mu = pi + tau Sigma P' (P tau Sigma P' + Omega)^-1 (Q - P pi)
and posterior covariance Sigma + M with M = ((tau Sigma)^-1 + P' Omega^-1 P)^-1. With no views the
posterior mean is the prior exactly, and the unconstrained optimal weights are the market weights.
Omega defaults to He-Litterman's tau * diag(P Sigma P'); a `confidence` in (0, 1] scales it by (1/c - 1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class View:
    assets: Mapping[str, float]  # pick weights: {"A": 1} absolute, {"A": 1, "B": -1} relative
    ret: float  # annual (excess) return the view asserts for that combination
    confidence: float | None = None  # None -> He-Litterman omega; 1.0 -> near certainty


@dataclass
class BLResult:
    prior: pd.Series
    posterior_mean: pd.Series
    posterior_cov: pd.DataFrame
    P: pd.DataFrame
    Q: pd.Series
    omega: pd.DataFrame
    tau: float
    risk_aversion: float
    notes: list[str] = field(default_factory=list)


def market_risk_aversion(market_excess_return: float, market_variance: float) -> float:
    """delta = E[r_m - rf] / Var(r_m)."""
    if market_variance <= 0:
        raise ValueError("market variance must be positive")
    return float(market_excess_return / market_variance)


def implied_returns(
    cov: pd.DataFrame, market_weights: pd.Series | Mapping[str, float], risk_aversion: float = 2.5
) -> pd.Series:
    """pi = delta Sigma w: the excess returns for which `market_weights` are mean-variance optimal."""
    w = pd.Series(market_weights, dtype=float).reindex(cov.index)
    if w.isna().any():
        raise ValueError(f"market weights missing for {list(w[w.isna()].index)}")
    pi = risk_aversion * cov.to_numpy(dtype=float) @ w.to_numpy(dtype=float)
    return pd.Series(pi, index=cov.index, name="implied_return")


def _views_to_matrices(
    views: Sequence[View], cov: pd.DataFrame, tau: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    assets = list(cov.index)
    notes: list[str] = []
    P = np.zeros((len(views), len(assets)))
    Q = np.zeros(len(views))
    omega = np.zeros((len(views), len(views)))
    sigma = cov.to_numpy(dtype=float)
    for k, v in enumerate(views):
        unknown = [a for a in v.assets if a not in assets]
        if unknown:
            raise ValueError(f"view {k} references assets not in cov: {unknown}")
        for a, pick in v.assets.items():
            P[k, assets.index(a)] = float(pick)
        Q[k] = float(v.ret)
        base = float(tau * P[k] @ sigma @ P[k])
        if v.confidence is None:
            omega[k, k] = base
        else:
            if not 0.0 < v.confidence <= 1.0:
                raise ValueError("confidence must be in (0, 1]")
            omega[k, k] = max(base * (1.0 / v.confidence - 1.0), 1e-12 * max(base, 1e-12))
            notes.append(f"view {k}: confidence {v.confidence:g} -> omega {omega[k, k]:.3e}")
    labels = [f"view_{k}" for k in range(len(views))]
    return (
        pd.DataFrame(P, index=labels, columns=assets),
        pd.Series(Q, index=labels, name="Q"),
        pd.DataFrame(omega, index=labels, columns=labels),
        notes,
    )


def black_litterman(
    cov: pd.DataFrame,
    market_weights: pd.Series | Mapping[str, float],
    views: Sequence[View] = (),
    *,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    prior: pd.Series | Mapping[str, float] | None = None,
) -> BLResult:
    if tau <= 0:
        raise ValueError("tau must be positive")
    assets = list(cov.index)
    sigma = cov.to_numpy(dtype=float)
    pi = (
        pd.Series(prior, dtype=float).reindex(cov.index)
        if prior is not None
        else implied_returns(cov, market_weights, risk_aversion)
    )
    if pi.isna().any():
        raise ValueError("prior has missing values")
    notes: list[str] = []
    if not views:
        post_cov = pd.DataFrame(sigma * (1.0 + tau), index=assets, columns=assets)
        notes.append("no views: posterior mean equals the equilibrium prior")
        empty_p = pd.DataFrame(np.zeros((0, len(assets))), columns=assets)
        return BLResult(
            pi.rename("prior"),
            pi.rename("posterior_mean").copy(),
            post_cov,
            empty_p,
            pd.Series(dtype=float, name="Q"),
            pd.DataFrame(),
            tau,
            risk_aversion,
            notes,
        )
    P, Q, omega, vnotes = _views_to_matrices(views, cov, tau)
    notes += vnotes
    p, q, om = P.to_numpy(), Q.to_numpy(), omega.to_numpy()
    ts = tau * sigma
    middle = np.linalg.inv(p @ ts @ p.T + om)
    adj = ts @ p.T @ middle @ (q - p @ pi.to_numpy())
    post_mean = pd.Series(pi.to_numpy() + adj, index=assets, name="posterior_mean")
    m = ts - ts @ p.T @ middle @ p @ ts
    post_cov = pd.DataFrame(sigma + m, index=assets, columns=assets)
    return BLResult(pi.rename("prior"), post_mean, post_cov, P, Q, omega, tau, risk_aversion, notes)


def bl_weights(
    posterior_mean: pd.Series, cov: pd.DataFrame, risk_aversion: float = 2.5, *, normalise: bool = False
) -> pd.Series:
    """Unconstrained mean-variance weights w = (delta Sigma)^-1 mu; may be negative or lever up."""
    mu = pd.Series(posterior_mean, dtype=float).reindex(cov.index).to_numpy()
    w = np.linalg.solve(risk_aversion * cov.to_numpy(dtype=float), mu)
    if normalise and abs(w.sum()) > 1e-12:
        w = w / w.sum()
    return pd.Series(w, index=cov.index, name="weight")
