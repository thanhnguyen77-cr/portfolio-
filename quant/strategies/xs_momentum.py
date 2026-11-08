"""Cross-sectional momentum: rank by trailing return (skipping the most recent month), hold the top names.

Rebalances on the first session of each calendar month using as-of-adjusted closes up to the decision
session. Equal weight across the top `top_n` (or top `top_fraction`) instruments; everything else zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from quant.backtest.strategy import Context, Strategy


class CrossSectionalMomentum(Strategy):
    name = "xs_momentum"

    def __init__(
        self,
        instruments: Sequence[str],
        lookback: int = 252,
        skip: int = 21,
        top_n: int | None = None,
        top_fraction: float = 0.3,
        rebalance: str = "monthly",
    ) -> None:
        if lookback <= skip:
            raise ValueError("lookback must exceed skip")
        super().__init__(
            instruments,
            lookback=lookback,
            skip=skip,
            top_n=top_n,
            top_fraction=top_fraction,
            rebalance=rebalance,
        )
        self.lookback, self.skip, self.top_n, self.top_fraction, self.rebalance = (
            lookback,
            skip,
            top_n,
            top_fraction,
            rebalance,
        )
        self.warmup = lookback + 1
        self._last_period: tuple[int, int] | None = None

    def _is_rebalance_session(self, ctx: Context) -> bool:
        d = ctx.session_date
        period = (
            (d.year, d.month) if self.rebalance == "monthly" else (d.isocalendar()[0], d.isocalendar()[1])
        )
        if period == self._last_period:
            return False
        self._last_period = period
        return True

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        if not self._is_rebalance_session(ctx):
            return None
        scores: dict[str, float] = {}
        for iid in self.instruments:
            h = ctx.history(iid, self.lookback + 1)
            if len(h) < self.lookback + 1:
                continue
            px = h["adj_close"].to_numpy()
            past, recent = px[-(self.lookback + 1)], px[-(self.skip + 1)]
            if past > 0 and recent > 0:
                scores[iid] = recent / past - 1.0
        if not scores:
            return None
        k = self.top_n or max(1, math.ceil(len(scores) * self.top_fraction))
        winners = [iid for iid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        w = 1.0 / len(winners)
        return {iid: (w if iid in winners else 0.0) for iid in self.instruments}
