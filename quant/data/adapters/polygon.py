"""Polygon.io daily aggregates plus reference dividends and splits (US equities).

Needs QUANT_POLYGON_API_KEY. The free tier allows 5 requests per minute, hence the 12-second throttle.
Aggregates are requested with `adjusted=false` (stored as `raw`); the reference endpoints give ex-dates,
cash amounts and split ratios for the corporate_actions table.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.time import to_date
from quant.core.types import AdjustmentState, Instrument
from quant.data.adapters._bars import actions_frame, canonical_bars, empty_actions
from quant.data.adapters.base import (
    AdapterResult,
    BaseHTTPAdapter,
    Capability,
    FieldMap,
    RateLimit,
    empty_bars,
)

AGGS_URL = "https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
DIVIDENDS_URL = "https://api.polygon.io/v3/reference/dividends"
SPLITS_URL = "https://api.polygon.io/v3/reference/splits"


class PolygonAdapter(BaseHTTPAdapter):
    name = "polygon"
    identifier_key = "polygon"
    rate_limit = RateLimit(min_interval_s=12.0, requests_per_minute=5, note="free tier: 5 requests/min")
    field_map = FieldMap(
        open="o", high="h", low="l", close="c", volume="v", adj_close=None, dividends=None, splits=None
    )
    adjustment_state = AdjustmentState.RAW

    def __init__(
        self, api_key: str | None = None, session=None, drop_unfinished_session: bool = True
    ) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.api_key = api_key
        self.drop_unfinished_session = drop_unfinished_session

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BARS_DAILY, Capability.CORPORATE_ACTIONS})

    def _key(self) -> str:
        if not self.api_key:
            raise AdapterNotConfigured("Polygon needs QUANT_POLYGON_API_KEY")
        return self.api_key

    def _paged(self, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        key = self._key()
        for _ in range(100):
            body = self._http("GET", url, params={**params, "apiKey": key}).json()
            out.extend(body.get("results") or [])
            nxt = body.get("next_url")
            if not nxt:
                break
            url, params = nxt, {}
        return out

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        if freq != "1d":
            raise AdapterError(f"{self.name}: only daily bars are supported (got {freq!r})")
        start_d, end_d = to_date(start), to_date(end)
        self._key()
        frames, errors = [], {}
        for inst in instruments:
            sym = self.symbol_for(inst)
            url = AGGS_URL.format(symbol=sym, start=start_d.isoformat(), end=end_d.isoformat())
            try:
                rows = self._paged(url, {"adjusted": "false", "sort": "asc", "limit": 50000})
                bars = self.parse_aggs(inst, rows, start_d, end_d)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if bars.empty:
                errors[inst.instrument_id] = f"no aggregates for {sym} in {start_d}..{end_d}"
                continue
            frames.append(bars)
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_bars(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": start_d.isoformat(), "end": end_d.isoformat(), "adjusted": False},
        )

    def parse_aggs(
        self, inst: Instrument, rows: Sequence[dict[str, Any]], start: date, end: date
    ) -> pd.DataFrame:
        if not rows:
            return empty_bars()
        fm = self.field_map
        ts = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert("America/New_York")
        frame = pd.DataFrame(
            {
                "session_date": ts.tz_localize(None).normalize(),
                "open": [r.get(fm.open) for r in rows],
                "high": [r.get(fm.high) for r in rows],
                "low": [r.get(fm.low) for r in rows],
                "close": [r.get(fm.close) for r in rows],
                "volume": [r.get(fm.volume, 0.0) for r in rows],
            }
        )
        return canonical_bars(
            inst,
            frame,
            self.name,
            self.adjustment_state,
            drop_unfinished=self.drop_unfinished_session,
            start=start,
            end=end,
        )

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        self._key()
        frames, errors = [], {}
        for inst in instruments:
            sym = self.symbol_for(inst)
            try:
                divs = self._paged(
                    DIVIDENDS_URL,
                    {
                        "ticker": sym,
                        "ex_dividend_date.gte": start_d.isoformat(),
                        "ex_dividend_date.lte": end_d.isoformat(),
                        "limit": 1000,
                    },
                )
                splits = self._paged(
                    SPLITS_URL,
                    {
                        "ticker": sym,
                        "execution_date.gte": start_d.isoformat(),
                        "execution_date.lte": end_d.isoformat(),
                        "limit": 1000,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            frames.append(self.parse_actions(inst, divs, splits))
        frame = pd.concat(frames, ignore_index=True) if frames else empty_actions()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": start_d.isoformat(), "end": end_d.isoformat()},
        )

    def parse_actions(
        self, inst: Instrument, dividends: Sequence[dict[str, Any]], splits: Sequence[dict[str, Any]]
    ) -> pd.DataFrame:
        rows: list[tuple[str, object, str, float, str | None]] = []
        for d in dividends:
            amt = d.get("cash_amount")
            ex = d.get("ex_dividend_date")
            if amt and ex:
                ccy = str(d.get("currency") or inst.currency.value).upper()
                rows.append((inst.instrument_id, ex, "dividend", float(amt) * float(inst.price_scale), ccy))
        for s in splits:
            frm, to = s.get("split_from"), s.get("split_to")
            ex = s.get("execution_date")
            if frm and to and ex:
                rows.append((inst.instrument_id, ex, "split", float(to) / float(frm), None))
        return actions_frame(rows, self.name)
