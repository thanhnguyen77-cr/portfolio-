"""Long-only moving-average trend following, equal weight across instruments in an uptrend.

Signal on session t uses as-of-adjusted closes up to t: long when SMA(fast) > SMA(slow). Rebalances only
when the set of instruments in an uptrend changes (otherwise equal weights would drift-trade daily).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from quant.backtest.strategy import Context, Strategy
from quant.technical.trend import sma


class TrendFollowing(Strategy):
    name = "trend_following"

    def __init__(
        self,
        instruments: Sequence[str],
        fast: int = 50,
        slow: int = 200,
        rebalance_on_change_only: bool = True,
    ) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        super().__init__(instruments, fast=fast, slow=slow, rebalance_on_change_only=rebalance_on_change_only)
        self.fast, self.slow = fast, slow
        self.warmup = slow
        self.rebalance_on_change_only = rebalance_on_change_only
        self._last_longs: frozenset[str] | None = None

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        longs = []
        for iid in self.instruments:
            h = ctx.history(iid, self.slow)
            if len(h) < self.slow:
                continue
            px = h["adj_close"].reset_index(drop=True)
            if sma(px, self.fast).iloc[-1] > sma(px, self.slow).iloc[-1]:
                longs.append(iid)
        current = frozenset(longs)
        if self.rebalance_on_change_only and current == self._last_longs:
            return None
        self._last_longs = current
        w = 1.0 / len(longs) if longs else 0.0
        return {iid: (w if iid in current else 0.0) for iid in self.instruments}
