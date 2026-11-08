"""Quality / value / growth scores as cross-sectional percentile ranks (0-100).

Each component metric is rank-transformed across the universe with its sign convention (higher-is-better or
lower-is-better) and the score is the mean of the available component percentiles x 100. The number of
metrics each score used is reported, because a score built on one metric is weaker than one built on five.
These are descriptive rankings, not forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUALITY: dict[str, int] = {
    "roic": 1,
    "roe": 1,
    "gross_margin": 1,
    "fcf_margin": 1,
    "accruals": -1,
    "debt_to_equity": -1,
}
VALUE: dict[str, int] = {"earnings_yield": 1, "fcf_yield": 1, "pb": -1, "ev_ebitda": -1, "ps": -1}
GROWTH: dict[str, int] = {"revenue_growth": 1, "eps_growth": 1, "net_income_growth": 1}


def _component(frame: pd.DataFrame, spec: dict[str, int]) -> tuple[pd.Series, pd.Series]:
    ranks = []
    for metric, sign in spec.items():
        if metric not in frame.columns:
            continue
        col = pd.to_numeric(frame[metric], errors="coerce")
        if col.notna().sum() < 2:
            continue
        pct = col.rank(pct=True)
        ranks.append(pct if sign > 0 else 1.0 - pct + 1.0 / max(col.notna().sum(), 1))
    if not ranks:
        return pd.Series(np.nan, index=frame.index), pd.Series(0, index=frame.index)
    stack = pd.concat(ranks, axis=1)
    return stack.mean(axis=1, skipna=True) * 100.0, stack.notna().sum(axis=1)


def score_cross_section(ratios: pd.DataFrame) -> pd.DataFrame:
    """ratios: index = instrument_id, columns = ratio names (from compute_ratios)."""
    out = pd.DataFrame(index=ratios.index)
    for name, spec in (("quality", QUALITY), ("value", VALUE), ("growth", GROWTH)):
        score, n = _component(ratios, spec)
        out[f"{name}_score"] = score.round(1)
        out[f"{name}_n"] = n
    components = out[["quality_score", "value_score", "growth_score"]]
    out["composite"] = components.mean(axis=1, skipna=True).round(1)
    return out.sort_values("composite", ascending=False)
