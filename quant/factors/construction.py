"""Factor characteristics and cross-sectional normalisation.

Price-based characteristics use only data up to the scoring date. Fundamental characteristics come from
`compute_ratios` on a point-in-time StatementSet (`as_of` = scoring date), so a quarter cannot enter a
score before it was filed. Scores are rank-normalised across the cross-section (optionally within sector),
mapped to standard-normal quantiles, and combined with the sign conventions in FACTOR_SIGNS so that a
higher score always means "more of the factor" (cheaper, stronger momentum, higher quality, lower vol,
smaller).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

FACTOR_SIGNS: dict[str, dict[str, int]] = {
    "value": {"earnings_yield": 1, "fcf_yield": 1, "pb": -1},
    "quality": {"roic": 1, "gross_margin": 1, "accruals": -1, "debt_to_equity": -1},
    "momentum": {"mom_12_1": 1},
    "low_vol": {"vol_252": -1},
    "size": {"log_mcap": -1},  # long small, short big (SMB convention)
}


def price_characteristics(
    closes: pd.DataFrame,
    as_of: date | pd.Timestamp,
    *,
    lookback: int = 252,
    skip: int = 21,
    vol_window: int = 252,
    shares: Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """mom_12_1, vol_252 (annualised) and, with shares, log_mcap for every column of `closes` as of a date.

    `closes` is a wide frame (dates x instrument_id) of adjusted closes; only rows <= as_of are used.
    """
    hist = closes[closes.index <= pd.Timestamp(as_of)]
    out = pd.DataFrame(index=closes.columns)
    if len(hist) > lookback:
        out["mom_12_1"] = hist.iloc[-1 - skip] / hist.iloc[-1 - lookback] - 1.0
    else:
        out["mom_12_1"] = np.nan
    logret = np.log(hist).diff().iloc[-vol_window:]
    out["vol_252"] = logret.std(ddof=1) * np.sqrt(252) if len(logret) >= max(20, vol_window // 4) else np.nan
    if shares is not None:
        sh = pd.Series(shares, dtype="float64").reindex(closes.columns)
        out["log_mcap"] = np.log(hist.iloc[-1] * sh)
    out["price"] = hist.iloc[-1] if len(hist) else np.nan
    return out


def fundamental_characteristics(
    instrument_ids: Sequence[str],
    as_of: date,
    ratios_fn: Callable[[str, date], Mapping[str, float | None] | None],
    metrics: Sequence[str] = (
        "earnings_yield",
        "fcf_yield",
        "pb",
        "roic",
        "gross_margin",
        "accruals",
        "debt_to_equity",
    ),
) -> pd.DataFrame:
    """PIT ratio characteristics; `ratios_fn(instrument_id, as_of)` must only use data filed by then."""
    rows = {}
    for iid in instrument_ids:
        r = ratios_fn(iid, as_of)
        rows[iid] = {m: (None if r is None else r.get(m)) for m in metrics}
    return pd.DataFrame.from_dict(rows, orient="index").astype("float64")


def rank_normalise(
    raw: pd.DataFrame,
    *,
    groups: pd.Series | Mapping[str, str] | None = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """Map each column to standard-normal quantiles of its cross-sectional rank (optionally within groups).

    NaNs stay NaN. Groups smaller than `min_group_size` fall back to the whole cross-section.
    """

    def _z(col: pd.Series) -> pd.Series:
        valid = col.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=col.index)
        pct = valid.rank(method="average") / (len(valid) + 1.0)
        return pd.Series(stats.norm.ppf(pct), index=valid.index).reindex(col.index)

    out = pd.DataFrame(index=raw.index, columns=raw.columns, dtype="float64")
    if groups is None:
        for c in raw.columns:
            out[c] = _z(raw[c])
        return out
    grp = pd.Series(groups).reindex(raw.index)
    fallback = {c: _z(raw[c]) for c in raw.columns}
    for c in raw.columns:
        col = raw[c]
        res = fallback[c].copy()
        for idx in grp.groupby(grp).groups.values():
            members = [i for i in idx if i in col.index]
            if len(col.loc[members].dropna()) >= min_group_size:
                res.loc[members] = _z(col.loc[members]).to_numpy()
        out[c] = res
    return out


def composite_score(normalised: pd.DataFrame, spec: Mapping[str, int]) -> pd.Series:
    """Signed mean of the available normalised components for one factor (NaN when none available)."""
    parts = [normalised[m] * sign for m, sign in spec.items() if m in normalised.columns]
    if not parts:
        return pd.Series(np.nan, index=normalised.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def factor_scores(
    normalised: pd.DataFrame,
    factors: Sequence[str] = tuple(FACTOR_SIGNS),
    signs: Mapping[str, Mapping[str, int]] | None = None,
) -> pd.DataFrame:
    """One standardised score column per factor from a normalised characteristic frame."""
    table = signs or FACTOR_SIGNS
    out = pd.DataFrame(index=normalised.index)
    for f in factors:
        out[f] = composite_score(normalised, table[f])
    return out
