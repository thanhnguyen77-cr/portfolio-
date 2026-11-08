"""Buy and hold: move to the target weights on the first decision session, then never trade again."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from quant.backtest.strategy import Context, Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def __init__(self, instruments: Sequence[str], weights: Mapping[str, float] | None = None) -> None:
        super().__init__(instruments, weights=dict(weights) if weights else None)
        n = len(self.instruments)
        self.weights = dict(weights) if weights else {iid: 1.0 / n for iid in self.instruments}
        self._done = False

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        if self._done:
            return None
        if any(ctx.close(iid) is None for iid in self.instruments):
            return None  # wait until every instrument has traded at least once
        self._done = True
        return self.weights
