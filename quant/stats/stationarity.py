"""Unit-root and stationarity tests with one result shape.

ADF / Phillips-Perron: H0 = unit root (reject => stationary). KPSS: H0 = stationary (reject => unit root).
Variance ratio: H0 = random walk. `stationarity_summary` runs all four and states the agreement.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    pvalue: float
    lags: int | None
    nobs: int
    critical_values: dict[str, float]
    null: str
    reject_at_5pct: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _clean(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(pd.Series(x).dropna(), dtype="float64")
    if len(arr) < 20:
        raise ValueError("need at least 20 observations")
    return arr


def adf_test(x: pd.Series | np.ndarray, regression: str = "c", autolag: str | None = "AIC") -> TestResult:
    from statsmodels.tsa.stattools import adfuller

    arr = _clean(x)
    try:
        res = adfuller(arr, regression=regression, autolag=autolag, result_object=False)
    except TypeError:  # statsmodels < 0.15
        res = adfuller(arr, regression=regression, autolag=autolag)
    stat, p, lags, nobs, crit = res[0], res[1], res[2], res[3], res[4]
    return TestResult(
        "adf",
        float(stat),
        float(p),
        int(lags),
        int(nobs),
        {k: float(v) for k, v in crit.items()},
        "unit root",
        bool(p < 0.05),
    )


def kpss_test(x: pd.Series | np.ndarray, regression: str = "c", nlags: str | int = "auto") -> TestResult:
    from statsmodels.tsa.stattools import kpss

    arr = _clean(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # p-value is interpolated/truncated at the table bounds
        stat, p, lags, crit = kpss(arr, regression=regression, nlags=nlags)
    return TestResult(
        "kpss",
        float(stat),
        float(p),
        int(lags),
        len(arr),
        {k: float(v) for k, v in crit.items()},
        "stationary",
        bool(p < 0.05),
    )


def phillips_perron(x: pd.Series | np.ndarray, trend: str = "c") -> TestResult:
    from arch.unitroot import PhillipsPerron

    arr = _clean(x)
    pp = PhillipsPerron(arr, trend=trend)
    return TestResult(
        "phillips_perron",
        float(pp.stat),
        float(pp.pvalue),
        int(pp.lags),
        int(pp.nobs),
        {k: float(v) for k, v in pp.critical_values.items()},
        "unit root",
        bool(pp.pvalue < 0.05),
    )


def variance_ratio(x: pd.Series | np.ndarray, lags: int = 2, robust: bool = True) -> TestResult:
    from arch.unitroot import VarianceRatio

    arr = _clean(x)
    vr = VarianceRatio(arr, lags=lags, robust=robust)
    return TestResult(
        "variance_ratio",
        float(vr.stat),
        float(vr.pvalue),
        lags,
        int(vr.nobs),
        {k: float(v) for k, v in vr.critical_values.items()},
        "random walk",
        bool(vr.pvalue < 0.05),
    )


def stationarity_summary(x: pd.Series | np.ndarray) -> dict[str, object]:
    """All four tests plus a verdict: 'stationary', 'unit_root' or 'inconclusive'."""
    adf, kp, pp = adf_test(x), kpss_test(x), phillips_perron(x)
    try:
        vr: TestResult | None = variance_ratio(x)
    except Exception:  # noqa: BLE001 - short samples
        vr = None
    votes_stationary = int(adf.reject_at_5pct) + int(pp.reject_at_5pct) + int(not kp.reject_at_5pct)
    verdict = (
        "stationary" if votes_stationary == 3 else ("unit_root" if votes_stationary == 0 else "inconclusive")
    )
    out = {"verdict": verdict, "adf": adf.as_dict(), "kpss": kp.as_dict(), "phillips_perron": pp.as_dict()}
    if vr is not None:
        out["variance_ratio"] = vr.as_dict()
    return out
