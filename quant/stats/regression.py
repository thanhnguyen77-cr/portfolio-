"""OLS with HAC standard errors and the standard diagnostics in one call."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegressionResult:
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    r2: float
    adj_r2: float
    nobs: int
    cov_type: str
    diagnostics: dict[str, float] = field(default_factory=dict)
    vif: pd.Series = field(default_factory=pd.Series)
    fitted: pd.Series = field(default_factory=pd.Series)
    resid: pd.Series = field(default_factory=pd.Series)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "se": self.bse, "t": self.tvalues, "p": self.pvalues})


def ols(
    y: pd.Series, X: pd.DataFrame | pd.Series, *, add_const: bool = True, hac_lags: int | None = None
) -> RegressionResult:  # noqa: N803
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import (
        acorr_breusch_godfrey,
        breaks_cusumolsresid,
        het_breuschpagan,
        het_white,
    )
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    xdf = X.to_frame() if isinstance(X, pd.Series) else X.copy()
    data = pd.concat([y.rename("__y"), xdf], axis=1).dropna()
    yy = data["__y"]
    xx = data.drop(columns="__y")
    design = sm.add_constant(xx, has_constant="add") if add_const else xx
    model = sm.OLS(yy, design)
    if hac_lags is not None:
        fit = model.fit(cov_type="HAC", cov_kwds={"maxlags": int(hac_lags)})
        cov_type = f"HAC({hac_lags})"
    else:
        fit = model.fit()
        cov_type = "nonrobust"
    resid = fit.resid
    diag: dict[str, float] = {"durbin_watson": float(durbin_watson(resid))}
    try:
        nlags = min(5, max(1, len(resid) // 20))
        try:
            bg = acorr_breusch_godfrey(fit, nlags=nlags, result_object=False)
        except TypeError:  # statsmodels < 0.15
            bg = acorr_breusch_godfrey(fit, nlags=nlags)
        diag["breusch_godfrey_stat"], diag["breusch_godfrey_p"] = float(bg[0]), float(bg[1])
    except Exception:  # noqa: BLE001
        pass
    try:
        bp = het_breuschpagan(resid, design)
        diag["breusch_pagan_stat"], diag["breusch_pagan_p"] = float(bp[0]), float(bp[1])
        wt = het_white(resid, design)
        diag["white_stat"], diag["white_p"] = float(wt[0]), float(wt[1])
    except Exception:  # noqa: BLE001
        pass
    jb = jarque_bera(resid)
    diag.update(
        {
            "jarque_bera_stat": float(jb[0]),
            "jarque_bera_p": float(jb[1]),
            "skew": float(jb[2]),
            "kurtosis": float(jb[3]),
        }
    )
    diag["condition_number"] = float(np.linalg.cond(np.asarray(design, dtype="float64")))
    try:
        cs = breaks_cusumolsresid(resid, ddof=design.shape[1])
        diag["cusum_stat"], diag["cusum_p"] = float(cs[0]), float(cs[1])
    except Exception:  # noqa: BLE001
        pass
    vif = pd.Series(dtype="float64")
    if xx.shape[1] >= 2:
        arr = np.asarray(design, dtype="float64")
        cols = list(design.columns)
        vif = pd.Series(
            {c: float(variance_inflation_factor(arr, i)) for i, c in enumerate(cols) if c != "const"}
        )
    return RegressionResult(
        fit.params,
        fit.bse,
        fit.tvalues,
        fit.pvalues,
        float(fit.rsquared),
        float(fit.rsquared_adj),
        int(fit.nobs),
        cov_type,
        diag,
        vif,
        fit.fittedvalues,
        resid,
    )
