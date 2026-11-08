"""Stooq daily bars: key-free CSV export, a fallback when Yahoo is unavailable.

Symbols: `identifiers["stooq"]` when present (e.g. `AAPL.US`, `VOD.UK`, `^SPX`), else `<symbol>.us`. Stooq
serves split-adjusted prices for equities; dividends are not backed out. Indices and FX (`gbpusd`) have no
volume. The export is best-effort and occasionally rate-limited by IP, in which case the body is a short
text rather than CSV and the instrument is reported as failed.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import date

import pandas as pd

from quant.core.errors import AdapterError
from quant.core.time import to_date
from quant.core.types import AdjustmentState, Instrument
from quant.data.adapters._bars import canonical_bars
from quant.data.adapters.base import (
    AdapterResult,
    BaseHTTPAdapter,
    Capability,
    FieldMap,
    RateLimit,
    empty_bars,
)

STOOQ_URL = "https://stooq.com/q/d/l/"


class StooqAdapter(BaseHTTPAdapter):
    name = "stooq"
    identifier_key = "stooq"
    rate_limit = RateLimit(min_interval_s=1.0, note="unofficial export; daily request quota per IP")
    field_map = FieldMap(
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        adj_close=None,
        dividends=None,
        splits=None,
    )
    adjustment_state = AdjustmentState.SPLIT_ADJUSTED

    def __init__(self, session=None, drop_unfinished_session: bool = True) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.drop_unfinished_session = drop_unfinished_session

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BARS_DAILY, Capability.FX})

    def symbol_for(self, inst: Instrument) -> str:
        if "stooq" in inst.identifiers:
            return inst.identifiers["stooq"].lower()
        return f"{inst.symbol.lower()}.us"

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        if freq != "1d":
            raise AdapterError(f"{self.name}: only daily bars are supported (got {freq!r})")
        start_d, end_d = to_date(start), to_date(end)
        frames, errors = [], {}
        for inst in instruments:
            sym = self.symbol_for(inst)
            params = {"s": sym, "i": "d", "d1": start_d.strftime("%Y%m%d"), "d2": end_d.strftime("%Y%m%d")}
            try:
                text = self._http("GET", STOOQ_URL, params=params).text
                bars = self.parse_csv(inst, text, start_d, end_d)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if bars.empty:
                errors[inst.instrument_id] = f"no data for {sym!r} in {start_d}..{end_d}"
                continue
            frames.append(bars)
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_bars(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": start_d.isoformat(), "end": end_d.isoformat(), "freq": freq},
        )

    def parse_csv(self, inst: Instrument, text: str, start: date, end: date) -> pd.DataFrame:
        head = text.lstrip()[:200].lower()
        if not head.startswith("date"):
            raise AdapterError(f"stooq returned no CSV for {inst.instrument_id}: {text.strip()[:80]!r}")
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip() for c in raw.columns]
        fm = self.field_map
        frame = pd.DataFrame(
            {
                "session_date": pd.to_datetime(raw["Date"], errors="coerce"),
                "open": raw.get(fm.open),
                "high": raw.get(fm.high),
                "low": raw.get(fm.low),
                "close": raw.get(fm.close),
                "volume": raw.get(fm.volume, 0.0),
            }
        ).dropna(subset=["session_date"])
        return canonical_bars(
            inst,
            frame,
            self.name,
            self.adjustment_state,
            drop_unfinished=self.drop_unfinished_session,
            start=start,
            end=end,
        )
