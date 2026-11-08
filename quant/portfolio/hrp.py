"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Tree clustering on the correlation distance sqrt(0.5 (1 - rho)), quasi-diagonalisation (the dendrogram's
leaf order), then recursive bisection allocating between halves in inverse proportion to their
inverse-variance cluster variances. No matrix inversion, so it tolerates singular covariance matrices;
it ignores expected returns entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from quant.portfolio.covariance import cov_to_corr
from quant.portfolio.mean_variance import Constraints, OptimisationResult


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.clip(0.5 * (1.0 - corr.to_numpy(dtype=float)), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def cluster_order(dist: pd.DataFrame, method: str = "single") -> list[str]:
    """Leaf order of the hierarchical clustering (quasi-diagonalisation)."""
    if dist.shape[0] == 1:
        return [str(dist.index[0])]
    condensed = squareform(dist.to_numpy(dtype=float), checks=False)
    link = linkage(condensed, method=method)
    return [str(dist.index[i]) for i in leaves_list(link)]


def _cluster_variance(sigma: np.ndarray, idx: list[int]) -> float:
    sub = sigma[np.ix_(idx, idx)]
    ivp = 1.0 / np.clip(np.diag(sub), 1e-300, None)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: pd.DataFrame, linkage_method: str = "single") -> pd.Series:
    assets = [str(a) for a in cov.index]
    sigma = np.atleast_2d(cov.to_numpy(dtype=float))
    order = cluster_order(correlation_distance(cov_to_corr(cov)), linkage_method)
    pos = {a: i for i, a in enumerate(assets)}
    w = pd.Series(1.0, index=order, dtype=float)
    clusters: list[list[str]] = [order]
    while clusters:
        nxt: list[list[str]] = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            c1, c2 = c[:half], c[half:]
            v1 = _cluster_variance(sigma, [pos[a] for a in c1])
            v2 = _cluster_variance(sigma, [pos[a] for a in c2])
            alpha = 1.0 - v1 / (v1 + v2) if (v1 + v2) > 0 else 0.5
            w[c1] *= alpha
            w[c2] *= 1.0 - alpha
            nxt += [c1, c2]
        clusters = nxt
    return w.reindex(assets).rename("weight")


def hrp(cov: pd.DataFrame, linkage_method: str = "single") -> OptimisationResult:
    from quant.portfolio.risk_parity import risk_contributions

    weights = hrp_weights(cov, linkage_method)
    sigma = cov.to_numpy(dtype=float)
    w = weights.to_numpy()
    return OptimisationResult(
        weights=weights,
        volatility=float(np.sqrt(max(w @ sigma @ w, 0.0))),
        expected_return=None,
        sharpe=None,
        method="hrp",
        success=True,
        message=f"hierarchical risk parity ({linkage_method} linkage)",
        binding=Constraints().binding(weights),
        extras={
            "order": cluster_order(correlation_distance(cov_to_corr(cov)), linkage_method),
            "risk_contributions": risk_contributions(weights, cov),
        },
    )
