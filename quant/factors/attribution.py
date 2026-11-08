"""Return attribution: split each day's return into alpha, factor contributions (beta x factor return) and
residual, so cumulative (arithmetic) contributions sum exactly to the cumulative return."""

from __future__ import annotations

import pandas as pd

from quant.factors.exposures import ExposureResult, factor_exposures


def attribute_returns(
    returns: pd.Series,
    factors: pd.DataFrame,
    exposures: ExposureResult | None = None,
    *,
    rf: float | pd.Series | None = None,
) -> pd.DataFrame:
    """Daily contributions with columns per factor plus alpha, rf, residual and total; sums match exactly."""
    exp = exposures or factor_exposures(returns, factors, rf=rf)
    data = pd.concat([returns.rename("total"), factors], axis=1, join="inner").dropna()
    out = pd.DataFrame(index=data.index)
    for f in factors.columns:
        out[f] = exp.betas[f] * data[f]
    out["alpha"] = exp.alpha_daily
    if rf is None:
        rf_series = pd.Series(0.0, index=data.index)
    elif isinstance(rf, pd.Series):
        rf_series = rf.reindex(data.index).ffill().fillna(0.0)
    else:
        rf_series = pd.Series(float(rf) / 252.0, index=data.index)
    out["rf"] = rf_series
    out["residual"] = data["total"] - out[list(factors.columns)].sum(axis=1) - out["alpha"] - out["rf"]
    out["total"] = data["total"]
    return out


def attribution_summary(contrib: pd.DataFrame) -> pd.DataFrame:
    """Cumulative (arithmetic) contribution and share of total per component."""
    total = contrib["total"].sum()
    rows = []
    for c in contrib.columns:
        if c == "total":
            continue
        s = contrib[c].sum()
        rows.append({"component": c, "cumulative": s, "share_of_total": s / total if total else float("nan")})
    rows.append({"component": "total", "cumulative": total, "share_of_total": 1.0})
    return pd.DataFrame(rows).set_index("component")
