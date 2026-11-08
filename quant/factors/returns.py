"""Long-short quantile factor returns.

At each rebalance date the cross-section is scored with data up to that date, the top quantile is held
long and the bottom quantile short in equal dollar amounts, and the positions drift (buy-and-hold) until
the next rebalance. Returns are gross of costs unless `cost_bps` is given (charged on one-way turnover).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from quant.backtest.metrics import PERIODS_PER_YEAR


def quantile_weights(
    scores: pd.Series, n_quantiles: int = 10, *, long_short: bool = True, min_per_side: int = 2
) -> pd.Series:
    """+1/n on the top quantile and -1/n on the bottom (dollar neutral), or long-only top quantile."""
    s = scores.dropna()
    if len(s) < 2 * min_per_side:
        return pd.Series(dtype="float64")
    n = max(min_per_side, int(np.floor(len(s) / n_quantiles)))
    ranked = s.sort_values(ascending=False)
    longs = ranked.index[:n]
    w = pd.Series(0.0, index=s.index)
    w.loc[longs] = 1.0 / len(longs)
    if long_short:
        shorts = ranked.index[-n:]
        w.loc[shorts] -= 1.0 / len(shorts)
    return w[w != 0]


def rebalance_dates(index: pd.DatetimeIndex, rebalance: str = "monthly") -> list[pd.Timestamp]:
    idx = pd.DatetimeIndex(index)
    if rebalance == "monthly":
        key = idx.to_period("M")
    elif rebalance == "weekly":
        key = idx.to_period("W")
    elif rebalance == "quarterly":
        key = idx.to_period("Q")
    else:
        raise ValueError("rebalance must be monthly, weekly or quarterly")
    firsts = pd.Series(idx, index=key).groupby(level=0).first()
    return [pd.Timestamp(t) for t in firsts]


@dataclass
class FactorReturns:
    returns: pd.DataFrame  # daily factor returns (columns = factors)
    turnover: pd.DataFrame  # one-way turnover at each rebalance
    holdings: dict[str, dict[pd.Timestamp, pd.Series]] = field(
        default_factory=dict
    )  # factor -> date -> weights
    notes: list[str] = field(default_factory=list)

    def stats(self) -> pd.DataFrame:
        return spread_stats(self.returns)


def factor_returns(
    closes: pd.DataFrame,
    score_fn: Callable[[pd.Timestamp], pd.DataFrame],
    *,
    factors: Sequence[str],
    start: date | pd.Timestamp | None = None,
    end: date | pd.Timestamp | None = None,
    rebalance: str = "monthly",
    n_quantiles: int = 10,
    long_short: bool = True,
    cost_bps: float = 0.0,
) -> FactorReturns:
    """Daily long-short returns per factor from a wide adjusted-close panel and a scoring callback.

    `score_fn(t)` returns a DataFrame (index instrument_id, one column per factor) using data <= t only.
    Positions are set from scores at t and earn returns from t+1 onward until the next rebalance.
    """
    px = closes.sort_index()
    if start is not None:
        px = px[px.index >= pd.Timestamp(start)]
    if end is not None:
        px = px[px.index <= pd.Timestamp(end)]
    rets = px.pct_change()
    dates = rebalance_dates(px.index, rebalance)
    out = pd.DataFrame(0.0, index=px.index, columns=list(factors))
    turnover_rows: list[dict[str, object]] = []
    holdings: dict[str, dict[pd.Timestamp, pd.Series]] = {f: {} for f in factors}
    prev_w: dict[str, pd.Series] = {f: pd.Series(dtype="float64") for f in factors}
    notes: list[str] = []
    for i, t in enumerate(dates):
        next_t = dates[i + 1] if i + 1 < len(dates) else None
        window = rets[(rets.index > t) & ((rets.index <= next_t) if next_t is not None else True)]
        if window.empty:
            continue
        scores = score_fn(t)
        row: dict[str, object] = {"rebalance": t}
        for f in factors:
            if f not in scores.columns:
                continue
            w = quantile_weights(scores[f], n_quantiles, long_short=long_short)
            if w.empty:
                notes.append(f"{f}: not enough scored names on {t.date()}")
                continue
            holdings[f][t] = w
            turn = float((w.subtract(prev_w[f], fill_value=0.0)).abs().sum() / 2.0)
            row[f] = turn
            prev_w[f] = w
            # buy-and-hold drift within the period: value of each leg compounds separately
            cum = (1.0 + window[w.index].fillna(0.0)).cumprod()
            value = (cum * w).sum(axis=1)  # portfolio value relative to a 1-unit book (net 0 if long-short)
            gross_prev = (cum.shift(1).fillna(1.0) * w.abs()).sum(axis=1)
            daily = value.diff().fillna(value.iloc[0] - w.sum())
            # express as return on the gross book (sum of |w| = 2 for a long-short, 1 for long-only)
            period_ret = daily / gross_prev.replace(0.0, np.nan) * (w.abs().sum())
            if cost_bps and turn:
                period_ret.iloc[0] -= turn * 2 * cost_bps / 1e4
            out.loc[window.index, f] = period_ret.to_numpy()
        turnover_rows.append(row)
    first = min((min(h) for h in holdings.values() if h), default=None)
    if first is not None:
        out = out[out.index > first]
    out.index.name = "date"
    return FactorReturns(
        out,
        pd.DataFrame(turnover_rows).set_index("rebalance") if turnover_rows else pd.DataFrame(),
        holdings,
        notes,
    )


def spread_stats(returns: pd.DataFrame, hac_lags: int = 5) -> pd.DataFrame:
    """Annualised mean, vol, Sharpe, HAC t-stat and hit rate per factor."""
    from statsmodels.regression.linear_model import OLS

    rows = []
    for f in returns.columns:
        r = returns[f].dropna()
        if len(r) < 20:
            continue
        mean, sd = r.mean(), r.std(ddof=1)
        fit = OLS(r.to_numpy(), np.ones(len(r))).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        rows.append(
            {
                "factor": f,
                "ann_return": mean * PERIODS_PER_YEAR,
                "ann_vol": sd * np.sqrt(PERIODS_PER_YEAR),
                "sharpe": (mean / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else np.nan,
                "t_stat_hac": float(fit.tvalues[0]),
                "hit_rate": float((r > 0).mean()),
                "sessions": int(len(r)),
            }
        )
    return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()
