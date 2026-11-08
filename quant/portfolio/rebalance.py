"""Vectorised periodic rebalancing on the engine's conventions: decide at close t, fill at the next open.

This exists for speed inside grids and frontier sweeps, not as a second source of truth - the test suite
pins it to the event-driven engine on identical inputs (`test_rebalance_matches_engine`). Its scope is
deliberately narrow and must stay inside what that test covers: one currency, no transaction costs, no
corporate actions, no cash or leverage constraint (the caller keeps gross weight comfortably below 1, or
accepts that the engine would have trimmed buys), full fills at the session open.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.backtest.fills import round_quantity


@dataclass
class RebalanceResult:
    equity: pd.Series  # close-of-session equity
    holdings: pd.DataFrame  # quantities held at each close
    cash: pd.Series
    trades: pd.DataFrame  # fill_date, instrument_id, qty (signed), price
    weights: pd.DataFrame  # realised weights at each close

    def returns(self) -> pd.Series:
        return self.equity.pct_change().iloc[1:].rename("return")


def simulate_target_weights(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    targets: pd.DataFrame,
    initial_cash: float,
    *,
    lot_sizes: Mapping[str, int] | None = None,
    fractional: bool = True,
    min_notional: float = 0.0,
) -> RebalanceResult:
    """`targets` rows are decision sessions (index must be a subset of `closes.index`); columns are assets.

    On a decision session the desired quantity is w * equity / close (the engine's rule); the signed
    difference to the current holding is rounded to lots, dropped when below `min_notional`, and filled at
    the next session's open. A decision on the final session has no next open and is ignored, as in the
    engine. Assets with a missing close keep their quantity.
    """
    closes = pd.DataFrame(closes).sort_index().astype(float)
    assets = [str(c) for c in closes.columns]
    opens = pd.DataFrame(opens).reindex(index=closes.index, columns=assets).astype(float)
    tgt = pd.DataFrame(targets).reindex(columns=assets).fillna(0.0).astype(float)
    unknown = tgt.index.difference(closes.index)
    if len(unknown):
        raise ValueError(f"target dates not in the price index: {list(unknown[:3])}")
    lots = np.array([int((lot_sizes or {}).get(a, 1)) for a in assets])
    idx = closes.index
    n, k = len(idx), len(assets)
    close_ff = closes.ffill().to_numpy()
    open_np = opens.to_numpy()
    decide = {idx.get_loc(t) for t in tgt.index}
    q = np.zeros(k)
    cash = float(initial_cash)
    equity = np.empty(n)
    cash_path = np.empty(n)
    hold = np.empty((n, k))
    trades: list[dict[str, object]] = []
    pending: np.ndarray | None = None
    for i in range(n):
        if pending is not None:
            px = open_np[i]
            fill = np.where(np.isnan(px), 0.0, pending)
            cash -= float(np.nansum(fill * np.nan_to_num(px)))
            q = q + fill
            for j in np.flatnonzero(fill):
                trades.append(
                    {
                        "fill_date": idx[i],
                        "instrument_id": assets[j],
                        "qty": float(fill[j]),
                        "price": float(px[j]),
                    }
                )
            pending = None
        c = np.nan_to_num(close_ff[i])
        eq = cash + float(q @ c)
        equity[i], cash_path[i], hold[i] = eq, cash, q
        if i in decide and i < n - 1:
            w = tgt.loc[idx[i]].to_numpy()
            desired = np.where(c > 0, w * eq / np.where(c > 0, c, 1.0), q)
            delta = desired - q
            delta = np.array(
                [round_quantity(float(d), int(lot), fractional) for d, lot in zip(delta, lots, strict=True)]
            )
            delta = np.where(np.abs(delta) * c < min_notional, 0.0, delta)
            pending = delta if np.any(delta != 0) else None
    eq_s = pd.Series(equity, index=idx, name="equity")
    holdings = pd.DataFrame(hold, index=idx, columns=assets)
    weights = holdings.mul(np.nan_to_num(close_ff)).div(eq_s, axis=0)
    trades_df = pd.DataFrame(trades, columns=["fill_date", "instrument_id", "qty", "price"])
    return RebalanceResult(eq_s, holdings, pd.Series(cash_path, index=idx, name="cash"), trades_df, weights)


def periodic_targets(
    index: pd.DatetimeIndex, weights: Mapping[str, float], rebalance: str = "monthly"
) -> pd.DataFrame:
    """Constant target weights on the first session of each period of `index`."""
    from quant.factors.returns import rebalance_dates

    dates = rebalance_dates(pd.DatetimeIndex(index), rebalance)
    return pd.DataFrame([dict(weights)] * len(dates), index=pd.DatetimeIndex(dates))
