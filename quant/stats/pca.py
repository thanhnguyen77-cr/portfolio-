"""Principal components of a return panel, and the absorption ratio (Kritzman et al. 2011)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PCAResult:
    explained_variance_ratio: pd.Series
    eigenvalues: pd.Series
    loadings: pd.DataFrame  # assets x components
    scores: pd.DataFrame  # dates x components
    standardized: bool


def pca(returns: pd.DataFrame, n_components: int | None = None, standardize: bool = True) -> PCAResult:
    df = returns.dropna(how="any")
    if df.shape[0] < df.shape[1] + 2:
        raise ValueError("need more observations than assets")
    x = df.to_numpy(dtype="float64")
    x = x - x.mean(axis=0)
    if standardize:
        sd = x.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        x = x / sd
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    eig = s**2 / (len(x) - 1)
    k = n_components or len(eig)
    comps = [f"pc{i + 1}" for i in range(k)]
    evr = pd.Series(eig[:k] / eig.sum(), index=comps, name="explained_variance_ratio")
    loadings = pd.DataFrame(vt[:k].T, index=df.columns, columns=comps)
    scores = pd.DataFrame(x @ vt[:k].T, index=df.index, columns=comps)
    return PCAResult(evr, pd.Series(eig[:k], index=comps, name="eigenvalue"), loadings, scores, standardize)


def absorption_ratio(
    returns: pd.DataFrame, n_eigen: int | None = None, window: int | None = None
) -> pd.Series | float:
    """Share of total variance absorbed by the first `n_eigen` components (default: 1/5 of the assets)."""
    n_assets = returns.shape[1]
    k = n_eigen or max(1, int(round(n_assets / 5)))

    def _ar(frame: pd.DataFrame) -> float:
        cov = np.cov(frame.to_numpy(dtype="float64"), rowvar=False)
        eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
        total = eig.sum()
        return float(eig[:k].sum() / total) if total > 0 else float("nan")

    clean = returns.dropna(how="any")
    if window is None:
        return _ar(clean)
    out = pd.Series(np.nan, index=clean.index, name="absorption_ratio")
    for i in range(window, len(clean) + 1):
        out.iloc[i - 1] = _ar(clean.iloc[i - window : i])
    return out
