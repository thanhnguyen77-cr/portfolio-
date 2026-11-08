"""Factor tilt: rank the cross-section on price-based factor scores and hold the top names equal-weight.

Uses only data up to the decision session: `price_characteristics` (12-1 momentum, trailing volatility)
on as-of-adjusted closes, rank-normalised across the cross-section (optionally within sector), combined
with the sign conventions in `FACTOR_SIGNS` and averaged across the requested factors. Fundamental
factors (value, quality, size) need statements or share counts that a strategy Context does not carry, so
this reference strategy is deliberately limited to momentum and low volatility.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from quant.backtest.strategy import Context, Strategy
from quant.core.types import Instrument
from quant.factors.construction import factor_scores, price_characteristics, rank_normalise
from quant.strategies.fixed_weights import period_key

PRICE_FACTORS = ("momentum", "low_vol")


class FactorTilt(Strategy):
    name = "factor_tilt"

    def __init__(
        self,
        instruments: Sequence[str],
        factors: Sequence[str] = PRICE_FACTORS,
        top_fraction: float = 0.3,
        top_n: int | None = None,
        rebalance: str = "monthly",
        lookback: int = 252,
        skip: int = 21,
        vol_window: int = 252,
        sector_neutral: bool = False,
    ) -> None:
        factors = tuple(factors)
        bad = [f for f in factors if f not in PRICE_FACTORS]
        if bad:
            raise ValueError(f"factor_tilt supports price-based factors {PRICE_FACTORS}; got {bad}")
        if not factors:
            raise ValueError("at least one factor is required")
        if lookback <= skip:
            raise ValueError("lookback must exceed skip")
        period_key(pd.Timestamp("2000-01-03").date(), rebalance)
        super().__init__(
            instruments,
            factors=list(factors),
            top_fraction=top_fraction,
            top_n=top_n,
            rebalance=rebalance,
            lookback=lookback,
            skip=skip,
            vol_window=vol_window,
            sector_neutral=sector_neutral,
        )
        self.factors = factors
        self.top_fraction, self.top_n = top_fraction, top_n
        self.rebalance = rebalance
        self.lookback, self.skip, self.vol_window = lookback, skip, vol_window
        self.sector_neutral = sector_neutral
        self.warmup = lookback + 1
        self._last_key: tuple[int, ...] | None = None
        self._sectors: dict[str, str] = {}

    def on_start(self, instruments: Mapping[str, Instrument]) -> None:
        self._sectors = {iid: (inst.sector or "unknown") for iid, inst in instruments.items()}

    def scores(self, ctx: Context) -> pd.Series:
        closes: dict[str, pd.Series] = {}
        for iid in self.instruments:
            h = ctx.history(iid, self.lookback + 1)
            if len(h) < self.lookback + 1:
                continue
            closes[iid] = h.set_index("session_date")["adj_close"]
        if len(closes) < 2:
            return pd.Series(dtype=float)
        panel = pd.DataFrame(closes).sort_index().ffill()
        chars = price_characteristics(
            panel, ctx.session_date, lookback=self.lookback, skip=self.skip, vol_window=self.vol_window
        )
        groups = (
            {iid: self._sectors.get(iid, "unknown") for iid in chars.index} if self.sector_neutral else None
        )
        norm = rank_normalise(chars[["mom_12_1", "vol_252"]], groups=groups)
        table = factor_scores(norm, factors=self.factors)
        return table.mean(axis=1, skipna=True).dropna()

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        key = period_key(ctx.session_date, self.rebalance)
        if key == self._last_key:
            return None
        composite = self.scores(ctx)
        if composite.empty:
            return None
        self._last_key = key
        k = self.top_n or max(1, math.ceil(len(composite) * self.top_fraction))
        winners = set(composite.nlargest(k).index)
        w = 1.0 / len(winners)
        return {iid: (w if iid in winners else 0.0) for iid in self.instruments}
