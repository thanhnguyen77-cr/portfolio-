"""Comparable-company valuation: apply peer multiples (median by default) to the target's metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# multiple -> (target metric, basis): 'equity' multiplies to an equity value, 'ev' to an enterprise value
MULTIPLES: dict[str, tuple[str, str]] = {
    "pe": ("net_income", "equity"),
    "pb": ("total_equity", "equity"),
    "ps": ("revenue", "equity"),
    "ev_ebitda": ("ebitda", "ev"),
    "ev_sales": ("revenue", "ev"),
}


def peer_multiples(ratios_by_peer: Mapping[str, Mapping[str, float | None]]) -> pd.DataFrame:
    """Rows = peers, columns = the multiples in MULTIPLES (NaN when a peer lacks one)."""
    rows = {peer: {m: r.get(m) for m in MULTIPLES} for peer, r in ratios_by_peer.items()}
    return pd.DataFrame.from_dict(rows, orient="index").astype("float64")


def comparable_valuation(
    target: Mapping[str, float | None],
    peers: pd.DataFrame,
    *,
    statistic: str = "median",
    trim: float = 0.0,
) -> pd.DataFrame:
    """One row per usable multiple: peer statistic, target metric, implied equity value and per-share value.

    `target` needs the metrics named in MULTIPLES plus `net_debt` and `shares` (per-share values need shares).
    `trim` drops that fraction of peers from each tail before taking the statistic (0.1 = 10% each side).
    """
    if statistic not in {"median", "mean"}:
        raise ValueError("statistic must be 'median' or 'mean'")
    net_debt = float(target.get("net_debt") or 0.0)
    shares = target.get("shares")
    rows = []
    for mult, (metric, basis) in MULTIPLES.items():
        if mult not in peers.columns:
            continue
        col = peers[mult].dropna()
        col = col[col > 0]
        if trim and len(col) >= 5:
            lo, hi = col.quantile(trim), col.quantile(1 - trim)
            col = col[(col >= lo) & (col <= hi)]
        value = target.get(metric)
        if col.empty or value is None or not np.isfinite(value) or value <= 0:
            continue
        peer_stat = float(col.median() if statistic == "median" else col.mean())
        implied = peer_stat * float(value)
        equity_value = implied - net_debt if basis == "ev" else implied
        rows.append(
            {
                "multiple": mult,
                "peers_used": int(len(col)),
                "peer_multiple": peer_stat,
                "target_metric": float(value),
                "implied_equity_value": equity_value,
                "implied_per_share": equity_value / shares if shares else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    summary = {
        "multiple": "median_of_multiples",
        "peers_used": int(out["peers_used"].min()),
        "peer_multiple": np.nan,
        "target_metric": np.nan,
        "implied_equity_value": float(out["implied_equity_value"].median()),
        "implied_per_share": float(out["implied_per_share"].median()) if shares else np.nan,
    }
    return pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
