"""Corporate actions and as-of price adjustment.

Prices are stored exactly as the vendor delivered them. To compare prices across a split or a dividend,
callers ask for an *adjusted* series **as of a date**: only actions with `ex_date <= as_of` are applied,
and only the kinds the stored series has not already absorbed (see `AdjustmentState`). Adjustment is
backward and multiplicative (CRSP-style): for an action on ex-date e, every bar strictly before e is
multiplied by

    split    : 1 / ratio                 ratio = new shares per old share
    dividend : 1 - D / close(e - 1)      D in the same currency as the prices (pounds, not pence)

so adjusted prices are continuous through the event and the latest bar is unchanged.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant.core.time import to_date
from quant.core.types import AdjustmentState

ADJ_COLUMNS = ("adj_factor", "adj_open", "adj_high", "adj_low", "adj_close")


def adjustment_factors(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    state: AdjustmentState | str,
    as_of: date | str | pd.Timestamp | None = None,
) -> pd.Series:
    """Per-bar multiplicative factor turning stored prices into as-of-adjusted prices.

    `bars` is one instrument's frame (sorted or not); `actions` has columns ex_date, action_type, value.
    """
    state = AdjustmentState(state)
    order = np.argsort(bars["session_date"].to_numpy(), kind="mergesort")
    dates = bars["session_date"].to_numpy()[order].astype("datetime64[ns]")
    closes = bars["close"].to_numpy(dtype="float64")[order]
    factors = np.ones(len(bars), dtype="float64")

    if len(bars) and not actions.empty and state is not AdjustmentState.FULLY_ADJUSTED:
        acts = actions.copy()
        acts["ex_date"] = pd.to_datetime(acts["ex_date"]).astype("datetime64[ns]")
        if as_of is not None:
            acts = acts[acts["ex_date"] <= pd.Timestamp(to_date(as_of))]
        if state is AdjustmentState.SPLIT_ADJUSTED:
            acts = acts[acts["action_type"] != "split"]
        for ex_date, kind, value in acts[["ex_date", "action_type", "value"]].itertuples(index=False):
            idx = int(np.searchsorted(dates, np.datetime64(ex_date, "ns"), side="left"))
            if idx == 0:
                continue  # no bars before the ex-date: nothing to adjust
            if kind == "split":
                if value > 0:
                    factors[:idx] *= 1.0 / float(value)
            elif kind == "dividend":
                prev_close = closes[idx - 1]
                if prev_close > 0 and 0 < value < prev_close:
                    factors[:idx] *= 1.0 - float(value) / prev_close
    out = np.empty_like(factors)
    out[order] = factors
    return pd.Series(out, index=bars.index, name="adj_factor")


def adjust_bars(
    bars: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    as_of: date | str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add adj_factor and adj_open/high/low/close to a (multi-instrument) bars frame."""
    if bars.empty:
        out = bars.copy()
        for col in ADJ_COLUMNS:
            out[col] = pd.Series(dtype="float64")
        return out
    pieces = []
    for iid, group in bars.groupby("instrument_id", sort=False):
        acts = actions[actions["instrument_id"] == iid] if not actions.empty else actions
        state = group["adjustment_state"].iloc[0]
        g = group.copy()
        g["adj_factor"] = adjustment_factors(g, acts, state=state, as_of=as_of)
        for col in ("open", "high", "low", "close"):
            g[f"adj_{col}"] = g[col] * g["adj_factor"]
        pieces.append(g)
    out = pd.concat(pieces).sort_values(["instrument_id", "session_date"], kind="mergesort")
    return out.reset_index(drop=True)


def split_ratios(actions: pd.DataFrame, instrument_id: str) -> pd.Series:
    """ex_date -> split ratio for one instrument (what a RAW-price engine must apply to quantities)."""
    if actions.empty:
        return pd.Series(dtype="float64")
    sel = actions[(actions["instrument_id"] == instrument_id) & (actions["action_type"] == "split")]
    index = pd.to_datetime(sel["ex_date"])
    return pd.Series(sel["value"].to_numpy(dtype="float64"), index=index, name="ratio")


def dividends(actions: pd.DataFrame, instrument_id: str) -> pd.DataFrame:
    """ex_date, amount, currency for one instrument's cash dividends."""
    if actions.empty:
        return pd.DataFrame({"ex_date": pd.Series(dtype="datetime64[ns]"), "amount": [], "currency": []})
    sel = actions[(actions["instrument_id"] == instrument_id) & (actions["action_type"] == "dividend")]
    return pd.DataFrame(
        {
            "ex_date": pd.to_datetime(sel["ex_date"]).to_numpy(),
            "amount": sel["value"].to_numpy(dtype="float64"),
            "currency": sel["currency"].to_numpy(),
        }
    )
