"""Fixed target weights, rebalanced on the first session of each period (or never)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

from quant.backtest.strategy import Context, Strategy


def period_key(d: date, rebalance: str) -> tuple[int, ...] | None:
    if rebalance == "monthly":
        return (d.year, d.month)
    if rebalance == "weekly":
        iso = d.isocalendar()
        return (iso[0], iso[1])
    if rebalance == "quarterly":
        return (d.year, (d.month - 1) // 3)
    if rebalance == "never":
        return None
    raise ValueError("rebalance must be monthly, weekly, quarterly or never")


class FixedWeights(Strategy):
    name = "fixed_weights"

    def __init__(
        self,
        instruments: Sequence[str],
        weights: Mapping[str, float] | None = None,
        rebalance: str = "monthly",
    ) -> None:
        period_key(date(2000, 1, 3), rebalance)  # validate early
        super().__init__(instruments, weights=dict(weights) if weights else None, rebalance=rebalance)
        n = len(self.instruments)
        self.weights = dict(weights) if weights else {iid: 1.0 / n for iid in self.instruments}
        self.rebalance = rebalance
        self._last_key: tuple[int, ...] | None = None
        self._done_once = False

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        if any(ctx.close(iid) is None for iid in self.instruments):
            return None
        key = period_key(ctx.session_date, self.rebalance)
        if key is None:
            if self._done_once:
                return None
            self._done_once = True
            return self.weights
        if key == self._last_key:
            return None
        self._last_key = key
        return self.weights
