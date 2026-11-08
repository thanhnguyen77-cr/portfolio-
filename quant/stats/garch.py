"""GARCH-family volatility models via `arch`. Returns are scaled to percent for estimation and the results
are converted back to decimal and annualised where it matters."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


@dataclass
class GarchResult:
    model: str
    dist: str
    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    conditional_vol: pd.Series  # annualised, decimal
    std_resid: pd.Series
    persistence: float
    converged: bool
    _fit: Any

    def summary(self) -> str:
        return str(self._fit.summary())


def fit_garch(
    returns: pd.Series,
    *,
    p: int = 1,
    q: int = 1,
    o: int = 0,
    model: str = "GARCH",
    dist: str = "t",
    mean: str = "Constant",
) -> GarchResult:
    """model: GARCH | GJR (sets o=1 asymmetry) | EGARCH; dist: normal | t | skewt."""
    from arch import arch_model

    r = returns.dropna().astype("float64") * 100.0
    if len(r) < 100:
        raise ValueError("need at least 100 returns to fit a GARCH model")
    vol = "GARCH"
    if model.upper() == "GJR":
        o = max(o, 1)
    elif model.upper() == "EGARCH":
        vol = "EGARCH"
    am = arch_model(r, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = am.fit(disp="off", show_warning=False)
    params = {k: float(v) for k, v in fit.params.items()}
    alpha = sum(v for k, v in params.items() if k.startswith("alpha"))
    beta = sum(v for k, v in params.items() if k.startswith("beta"))
    gamma = sum(v for k, v in params.items() if k.startswith("gamma"))
    persistence = alpha + beta + 0.5 * gamma if vol == "GARCH" else beta
    cond_vol = (fit.conditional_volatility / 100.0 * math.sqrt(PERIODS_PER_YEAR)).rename("cond_vol_annual")
    return GarchResult(
        model.upper(),
        dist,
        params,
        float(fit.loglikelihood),
        float(fit.aic),
        float(fit.bic),
        cond_vol,
        fit.std_resid.rename("std_resid"),
        float(persistence),
        bool(fit.convergence_flag == 0),
        fit,
    )


def forecast_vol(result: GarchResult, horizon: int = 10) -> pd.Series:
    """Annualised volatility forecast for each step ahead (decimal)."""
    fc = result._fit.forecast(horizon=horizon, reindex=False)
    var = np.asarray(fc.variance.iloc[-1], dtype="float64")
    return pd.Series(
        np.sqrt(var) / 100.0 * math.sqrt(PERIODS_PER_YEAR), index=range(1, horizon + 1), name="vol_forecast"
    )
