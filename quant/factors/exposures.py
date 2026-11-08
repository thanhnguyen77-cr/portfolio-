"""Time-series factor exposures: regress excess returns on factor returns with HAC standard errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.metrics import PERIODS_PER_YEAR
from quant.stats.regression import ols


@dataclass
class ExposureResult:
    alpha_daily: float
    alpha_annual: float
    alpha_t: float
    betas: pd.Series
    t_stats: pd.Series
    r2: float
    nobs: int
    cov_type: str
    residuals: pd.Series

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"beta": self.betas, "t_stat": self.t_stats})


def _excess(returns: pd.Series, rf: float | pd.Series | None) -> pd.Series:
    if rf is None:
        return returns
    if isinstance(rf, pd.Series):
        return returns - rf.reindex(returns.index).ffill().fillna(0.0)
    return returns - float(rf) / PERIODS_PER_YEAR


def factor_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    *,
    rf: float | pd.Series | None = None,
    hac_lags: int | None = 5,
) -> ExposureResult:
    """Alpha and betas of `returns` (daily) on `factors` (daily factor returns), excess of `rf` if given."""
    y = _excess(returns.dropna(), rf)
    data = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    if len(data) < len(factors.columns) + 10:
        raise ValueError("too few overlapping observations for an exposure regression")
    res = ols(data["y"], data[list(factors.columns)], hac_lags=hac_lags)
    alpha = float(res.params["const"])
    return ExposureResult(
        alpha_daily=alpha,
        alpha_annual=alpha * PERIODS_PER_YEAR,
        alpha_t=float(res.tvalues["const"]),
        betas=res.params.drop("const"),
        t_stats=res.tvalues.drop("const"),
        r2=res.r2,
        nobs=res.nobs,
        cov_type=res.cov_type,
        residuals=res.resid,
    )


def rolling_exposures(
    returns: pd.Series, factors: pd.DataFrame, window: int = 126, *, rf=None
) -> pd.DataFrame:  # noqa: ANN001
    """Rolling OLS betas (no HAC) over `window` sessions; index = window end date."""
    y = _excess(returns.dropna(), rf)
    data = pd.concat([y.rename("y"), factors], axis=1, join="inner").dropna()
    cols = list(factors.columns)
    rows = []
    for end in range(window, len(data) + 1):
        chunk = data.iloc[end - window : end]
        x = np.column_stack([np.ones(window), chunk[cols].to_numpy()])
        beta = np.linalg.lstsq(x, chunk["y"].to_numpy(), rcond=None)[0]
        rows.append(pd.Series(beta[1:], index=cols, name=chunk.index[-1]))
    return pd.DataFrame(rows)
