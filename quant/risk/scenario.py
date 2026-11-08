"""Factor-based scenario analysis: estimate asset betas to factor returns, then propagate factor shocks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def estimate_betas(returns: pd.DataFrame, factors: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """OLS betas of each asset on the factor panel (assets x factors), with an `alpha` and `r2` column."""
    data = pd.concat([returns, factors], axis=1, join="inner").dropna()
    if len(data) < min_obs:
        raise ValueError(f"need at least {min_obs} overlapping observations, got {len(data)}")
    x = np.column_stack([np.ones(len(data)), data[list(factors.columns)].to_numpy(dtype=float)])
    rows = {}
    for a in returns.columns:
        y = data[a].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coef
        ss_res = float(((y - fitted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rows[a] = {
            "alpha": coef[0],
            **{f: coef[i + 1] for i, f in enumerate(factors.columns)},
            "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        }
    return pd.DataFrame(rows).T


def factor_scenario(
    weights: pd.Series | Mapping[str, float], betas: pd.DataFrame, factor_shocks: Mapping[str, float]
) -> dict[str, object]:
    """Portfolio return = sum_i w_i sum_k beta_ik shock_k, with per-asset and per-factor contributions."""
    w = pd.Series(weights, dtype=float)
    b = betas.reindex(w.index)
    missing_assets = [a for a in w.index if a not in betas.index]
    if missing_assets:
        raise ValueError(f"no betas for {missing_assets}")
    shocks = pd.Series(factor_shocks, dtype=float)
    unknown = [f for f in shocks.index if f not in b.columns]
    if unknown:
        raise ValueError(f"factors not in betas: {unknown}")
    asset_ret = b[shocks.index].to_numpy(dtype=float) @ shocks.to_numpy()
    by_asset = pd.Series(asset_ret * w.to_numpy(), index=w.index, name="pnl")
    by_factor = pd.Series(
        {f: float((w * b[f]).sum() * shocks[f]) for f in shocks.index}, name="pnl", dtype=float
    )
    return {"portfolio_return": float(by_asset.sum()), "by_asset": by_asset, "by_factor": by_factor}


def scenario_table(
    weights: pd.Series | Mapping[str, float],
    betas: pd.DataFrame,
    scenarios: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    rows = []
    for name, shocks in scenarios.items():
        res = factor_scenario(weights, betas, shocks)
        rows.append(
            {
                "scenario": name,
                "portfolio_return": res["portfolio_return"],
                **{f"factor_{k}": v for k, v in res["by_factor"].items()},
                **{f"asset_{k}": v for k, v in res["by_asset"].items()},
            }
        )
    return pd.DataFrame(rows)
