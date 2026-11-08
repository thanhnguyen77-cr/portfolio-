"""Cointegration pairs trading (dollar-neutral, long/short; requires allow_short).

Every `reestimate_every` sessions the hedge ratio is re-estimated by Engle-Granger on the trailing
`lookback` of as-of-adjusted closes; the pair is tradable only while the residual tests as cointegrated at
`max_pvalue`. The spread's rolling z-score drives entries (|z| > entry_z), exits (|z| < exit_z) and stops
(|z| > stop_z). Positions are equal-notional long/short halves of `gross` (the hedge ratio is used for the
spread, not for sizing; see notes in the stage report).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from quant.backtest.strategy import Context, Strategy
from quant.stats.cointegration import engle_granger, spread_zscore


class PairsCointegration(Strategy):
    name = "pairs_cointegration"

    def __init__(
        self,
        instruments: Sequence[str],
        lookback: int = 252,
        z_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_pvalue: float = 0.05,
        reestimate_every: int = 21,
        gross: float = 1.0,
    ) -> None:
        if len(instruments) != 2:
            raise ValueError("pairs strategy needs exactly two instruments (y, x)")
        super().__init__(
            instruments,
            lookback=lookback,
            z_window=z_window,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            max_pvalue=max_pvalue,
            reestimate_every=reestimate_every,
            gross=gross,
        )
        self.lookback, self.z_window = lookback, z_window
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.max_pvalue, self.reestimate_every, self.gross = max_pvalue, reestimate_every, gross
        self.warmup = lookback + 1
        self._hedge: float | None = None
        self._intercept = 0.0
        self._tradable = False
        self._last_fit = -(10**9)
        self._state = 0  # +1 long spread (long y / short x), -1 short spread, 0 flat

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        y_id, x_id = self.instruments
        hy, hx = ctx.history(y_id, self.lookback), ctx.history(x_id, self.lookback)
        if len(hy) < self.lookback or len(hx) < self.lookback:
            return None
        py = hy.set_index("session_date")["adj_close"]
        px = hx.set_index("session_date")["adj_close"]
        if ctx.session_index - self._last_fit >= self.reestimate_every:
            try:
                eg = engle_granger(py, px)
                self._hedge, self._intercept = eg.hedge_ratio, eg.intercept
                self._tradable = (
                    eg.cointegrated_at_5pct if self.max_pvalue == 0.05 else eg.pvalue < self.max_pvalue
                )
            except ValueError:
                self._tradable = False
            self._last_fit = ctx.session_index
        if self._hedge is None:
            return None
        spread = py - self._hedge * px - self._intercept
        z = spread_zscore(spread, self.z_window).iloc[-1]
        if not np.isfinite(z):
            return None
        target = self._state
        if self._state == 0:
            if self._tradable and z > self.entry_z:
                target = -1
            elif self._tradable and z < -self.entry_z:
                target = 1
        else:
            if abs(z) < self.exit_z or abs(z) > self.stop_z or not self._tradable:
                target = 0
        if target == self._state:
            return None
        self._state = target
        half = self.gross / 2.0
        return {y_id: target * half, x_id: -target * half}
