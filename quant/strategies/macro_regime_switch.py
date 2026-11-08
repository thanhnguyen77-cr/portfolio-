"""Regime switch: hold a risk-on book while the benchmark's rule-based regime is `risk_on`, scale it down
in `caution`, and move to the risk-off book (default: cash) in `risk_off`.

The regime comes from `quant.technical.regime.rule_based_regime` on the benchmark's as-of-adjusted closes
(and an optional volatility index), so the benchmark and the volatility index must be in the strategy's
instrument list (with zero weight) for their history to be available. Rebalances when the label changes
or at the start of each period, whichever comes first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant.backtest.strategy import Context, Strategy
from quant.strategies.fixed_weights import period_key
from quant.technical.regime import rule_based_regime


class MacroRegimeSwitch(Strategy):
    name = "macro_regime_switch"

    def __init__(
        self,
        instruments: Sequence[str],
        benchmark: str,
        risk_on: Mapping[str, float],
        risk_off: Mapping[str, float] | None = None,
        vol_index: str | None = None,
        caution_scale: float = 0.5,
        rebalance: str = "weekly",
        history: int = 400,
    ) -> None:
        instruments = list(instruments)
        for ref, label in ((benchmark, "benchmark"), (vol_index, "vol_index")):
            if ref is not None and ref not in instruments:
                raise ValueError(f"{label} {ref!r} must be in the strategy's instruments (weight 0)")
        unknown = [k for k in {**dict(risk_on), **dict(risk_off or {})} if k not in instruments]
        if unknown:
            raise ValueError(f"weights reference instruments outside the list: {unknown}")
        if not 0.0 <= caution_scale <= 1.0:
            raise ValueError("caution_scale must be in [0, 1]")
        period_key(pd.Timestamp("2000-01-03").date(), rebalance)
        super().__init__(
            instruments,
            benchmark=benchmark,
            risk_on=dict(risk_on),
            risk_off=dict(risk_off or {}),
            vol_index=vol_index,
            caution_scale=caution_scale,
            rebalance=rebalance,
        )
        self.benchmark, self.vol_index = benchmark, vol_index
        self.risk_on, self.risk_off = dict(risk_on), dict(risk_off or {})
        self.caution_scale, self.rebalance, self.history = caution_scale, rebalance, history
        self.warmup = 200
        self._last_label: str | None = None
        self._last_key: tuple[int, ...] | None = None

    def regime(self, ctx: Context) -> str | None:
        h = ctx.history(self.benchmark, self.history)
        if len(h) < 200:
            return None
        bench = h.set_index("session_date")["adj_close"]
        vi = None
        if self.vol_index:
            hv = ctx.history(self.vol_index, self.history)
            if len(hv):
                vi = hv.set_index("session_date")["adj_close"]
        label = rule_based_regime(bench, vi)["regime"].iloc[-1]
        return None if label is None or (isinstance(label, float) and pd.isna(label)) else str(label)

    def weights_for(self, label: str) -> dict[str, float]:
        if label == "risk_on":
            book = dict(self.risk_on)
        elif label == "caution":
            book = {k: v * self.caution_scale for k, v in self.risk_on.items()}
            for k, v in self.risk_off.items():
                book[k] = book.get(k, 0.0) + v * (1.0 - self.caution_scale)
        else:
            book = dict(self.risk_off)
        return {iid: book.get(iid, 0.0) for iid in self.instruments}

    def on_session(self, ctx: Context) -> Mapping[str, float] | None:
        label = self.regime(ctx)
        if label is None:
            return None
        key = period_key(ctx.session_date, self.rebalance)
        if label == self._last_label and key == self._last_key:
            return None
        self._last_label, self._last_key = label, key
        return self.weights_for(label)
