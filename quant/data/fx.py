"""FX conversion built from cached FX bars.

Pair convention: symbol `GBPUSD` is the number of USD per 1 GBP, so `rate("GBP", "USD")` is its close,
`rate("USD", "GBP")` its reciprocal, and other crosses go through USD. Rates are aligned to requested
dates by **forward fill only**: a date before the first available fix stays NaN rather than borrowing
a future rate.

FIXING NOTE: vendor daily FX closes (Yahoo) are stamped ~17:00 New York, after the London (16:30) and
New York (16:00) equity closes. Converting equity closes at that fix is a small, documented approximation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import pandas as pd

from quant.core.errors import FXError
from quant.core.types import AssetClass, Instrument

FIXING_NOTE = (
    "FX rates are vendor daily closes (~17:00 New York); equity closes earlier in the day are converted at "
    "that fix. Missing FX dates are forward-filled, never back-filled."
)


class FXRates:
    """Rates keyed by pair symbol (e.g. 'GBPUSD') -> Series indexed by tz-naive session date."""

    def __init__(self, pairs: Mapping[str, pd.Series]) -> None:
        self._pairs: dict[str, pd.Series] = {}
        for sym, s in pairs.items():
            sym = sym.upper().replace("/", "")
            if len(sym) != 6:
                raise FXError(f"pair symbol {sym!r} must be two ISO codes, e.g. GBPUSD")
            series = pd.Series(s, dtype="float64")
            series.index = pd.DatetimeIndex(series.index).normalize()
            self._pairs[sym] = series[~series.index.duplicated(keep="last")].sort_index()

    @classmethod
    def from_bars(cls, bars: pd.DataFrame, instruments: Mapping[str, Instrument]) -> FXRates:
        """Build from a canonical bars frame holding FX instruments (uses each instrument's symbol)."""
        pairs: dict[str, pd.Series] = {}
        if bars.empty:
            return cls(pairs)
        for iid, g in bars.groupby("instrument_id"):
            inst = instruments.get(iid) if hasattr(instruments, "get") else instruments[iid]
            if inst is None or inst.asset_class is not AssetClass.FX:
                continue
            pairs[inst.symbol] = pd.Series(g["close"].to_numpy(), index=pd.DatetimeIndex(g["session_date"]))
        return cls(pairs)

    @property
    def pairs(self) -> list[str]:
        return sorted(self._pairs)

    def _raw(self, base: str, quote: str) -> pd.Series | None:
        if base + quote in self._pairs:
            return self._pairs[base + quote]
        if quote + base in self._pairs:
            return 1.0 / self._pairs[quote + base]
        return None

    def _series(self, base: str, quote: str) -> pd.Series:
        base, quote = base.upper(), quote.upper()
        direct = self._raw(base, quote)
        if direct is not None:
            return direct
        leg1, leg2 = self._raw(base, "USD"), self._raw("USD", quote)
        if leg1 is not None and leg2 is not None:
            idx = leg1.index.union(leg2.index)
            return (leg1.reindex(idx).ffill() * leg2.reindex(idx).ffill()).dropna()
        raise FXError(f"no FX rate for {base}->{quote}; available pairs: {self.pairs}")

    def rate(self, base: str, quote: str, dates: Sequence[date] | pd.DatetimeIndex) -> pd.Series:
        """Rate on each date (forward-filled; NaN before the first available fix)."""
        idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
        if base.upper() == quote.upper():
            return pd.Series(1.0, index=idx, name=f"{base.upper()}{quote.upper()}")
        s = self._series(base, quote)
        aligned = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        aligned.name = f"{base.upper()}{quote.upper()}"
        return aligned

    def convert(
        self, values: pd.Series, from_ccy: str, to_ccy: str, *, allow_missing: bool = False
    ) -> pd.Series:
        """Convert a date-indexed amount series; raises FXError on dates with no rate unless allowed."""
        rates = self.rate(from_ccy, to_ccy, pd.DatetimeIndex(values.index))
        if not allow_missing and rates.isna().any():
            first = rates[rates.isna()].index[0].date()
            raise FXError(f"no {from_ccy}->{to_ccy} rate on or before {first}")
        converted = values.to_numpy(dtype="float64") * rates.to_numpy()
        return pd.Series(converted, index=values.index, name=values.name)

    def latest(self, base: str, quote: str, as_of: date | pd.Timestamp) -> float:
        value = self.rate(base, quote, [pd.Timestamp(as_of)]).iloc[0]
        if pd.isna(value):
            raise FXError(f"no {base}->{quote} rate on or before {pd.Timestamp(as_of).date()}")
        return float(value)
