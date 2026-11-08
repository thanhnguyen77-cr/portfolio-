"""Correlation structure and its breakdown: average pairwise correlation (static and rolling), calm-vs-stress
correlation, diversification ratio, and the absorption ratio from quant.stats.pca."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from quant.stats.pca import absorption_ratio


def _offdiag_mean(corr: np.ndarray) -> float:
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    return float((corr.sum() - np.trace(corr)) / (n * (n - 1)))


def average_correlation(returns: pd.DataFrame, window: int | None = None) -> float | pd.Series:
    """Mean off-diagonal correlation, static or rolling (NaN until `window` observations)."""
    df = pd.DataFrame(returns).dropna(how="any")
    if window is None:
        return _offdiag_mean(df.corr().to_numpy())
    out = pd.Series(np.nan, index=df.index, name="avg_correlation")
    x = df.to_numpy(dtype=float)
    for i in range(window, len(df) + 1):
        out.iloc[i - 1] = _offdiag_mean(np.corrcoef(x[i - window : i], rowvar=False))
    return out


def correlation_breakdown(
    returns: pd.DataFrame, reference: pd.Series | None = None, quantile: float = 0.10
) -> dict[str, object]:
    """Correlations in calm periods versus stress periods (reference return in its bottom `quantile`).

    Conditioning on the portfolio's OWN worst days (the default reference) biases the stress correlations
    downward for the idiosyncratic part (Boyer, Gibson and Loretan, 1999): a very negative sum with a small
    common component implies the pieces offset each other. Prefer an external stress marker such as a
    volatility index or a macro indicator as `reference`; the result carries this note either way.
    """
    df = pd.DataFrame(returns).dropna(how="any")
    ref = (reference.reindex(df.index) if reference is not None else df.mean(axis=1)).astype(float)
    cut = float(ref.quantile(quantile))
    stress_mask = ref <= cut
    calm, stress = df[~stress_mask], df[stress_mask]
    if len(stress) < 5 or len(calm) < 5:
        raise ValueError("too few observations to split into calm and stress regimes")
    c_calm, c_stress = calm.corr(), stress.corr()
    return {
        "calm_corr": c_calm,
        "stress_corr": c_stress,
        "difference": c_stress - c_calm,
        "avg_calm": _offdiag_mean(c_calm.to_numpy()),
        "avg_stress": _offdiag_mean(c_stress.to_numpy()),
        "n_calm": int(len(calm)),
        "n_stress": int(len(stress)),
        "threshold": cut,
        "note": (
            "stress days selected by the portfolio's own returns: conditional correlations are biased "
            "(Boyer-Gibson-Loretan); use an external marker as `reference` where possible"
            if reference is None
            else "stress days selected by an external reference series"
        ),
    }


def diversification_ratio(weights: pd.Series | Mapping[str, float], cov: pd.DataFrame) -> float:
    """sum(w_i sigma_i) / sigma_p: 1 = no diversification benefit; sqrt(n) for n equal uncorrelated names."""
    w = pd.Series(weights, dtype=float).reindex(cov.index).fillna(0.0).to_numpy()
    sigma = cov.to_numpy(dtype=float)
    vols = np.sqrt(np.clip(np.diag(sigma), 0, None))
    port = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    return float((np.abs(w) * vols).sum() / port) if port > 0 else float("nan")


def rolling_absorption(returns: pd.DataFrame, window: int = 250, n_eigen: int | None = None) -> pd.Series:
    out = absorption_ratio(returns, n_eigen=n_eigen, window=window)
    assert isinstance(out, pd.Series)
    return out


def correlation_summary(
    returns: pd.DataFrame,
    cov: pd.DataFrame | None = None,
    weights: pd.Series | Mapping[str, float] | None = None,
) -> dict[str, float]:
    df = pd.DataFrame(returns).dropna(how="any")
    out: dict[str, float] = {"avg_correlation": float(average_correlation(df))}
    if df.shape[1] >= 2 and len(df) >= 50:
        br = correlation_breakdown(df)
        out["avg_correlation_calm"] = float(br["avg_calm"])
        out["avg_correlation_stress"] = float(br["avg_stress"])
        ar = absorption_ratio(df)
        out["absorption_ratio"] = float(ar) if not isinstance(ar, pd.Series) else float(ar.iloc[-1])
    if cov is not None and weights is not None:
        out["diversification_ratio"] = diversification_ratio(weights, cov)
    return out
