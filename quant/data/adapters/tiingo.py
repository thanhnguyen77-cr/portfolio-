"""Tiingo end-of-day prices (US equities and many ETFs) with dividends and split factors.

Needs QUANT_TIINGO_API_KEY (sent as `Authorization: Token ...`). Rows carry unadjusted open/high/low/close
plus `divCash` and `splitFactor` on the ex-date, so bars are stored `raw` and the actions table is filled
from the same response.
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

PRICES_URL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"


class TiingoAdapter(BaseHTTPAdapter):
    name = "tiingo"
    identifier_key = "tiingo"
    rate_limit = RateLimit(
        min_interval_s=0.2, requests_per_minute=50, note="free tier: 50 req/hour cap applies"
    )
    field_map = FieldMap(
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        adj_close="adjClose",
        dividends="divCash",
        splits="splitFactor",
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

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise AdapterNotConfigured("Tiingo needs QUANT_TIINGO_API_KEY")
        return {"Authorization": f"Token {self.api_key}", "Content-Type": "application/json"}

    def _fetch(self, inst: Instrument, start: date, end: date) -> list[dict[str, Any]]:
        url = PRICES_URL.format(symbol=self.symbol_for(inst))
        params = {
            "startDate": start.isoformat(),
            "endDate": end.isoformat(),
            "format": "json",
            "resampleFreq": "daily",
        }
        body = self._http("GET", url, params=params, headers=self._headers()).json()
        if isinstance(body, dict) and "detail" in body:
            raise AdapterError(f"tiingo: {body['detail']}")
        return list(body or [])

    def _batch(self, instruments: Sequence[Instrument], start: date, end: date, parse):  # noqa: ANN001,ANN202
        self._headers()  # credentials are checked once, loudly, not swallowed per instrument
        frames, errors = [], {}
        for inst in instruments:
            try:
                rows = self._fetch(inst, start, end)
                frame = parse(inst, rows, start, end)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if frame is None:
                errors[inst.instrument_id] = f"no rows for {self.symbol_for(inst)} in {start}..{end}"
                continue
            frames.append(frame)
        return frames, errors

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        if freq != "1d":
            raise AdapterError(f"{self.name}: only daily bars are supported (got {freq!r})")
        start_d, end_d = to_date(start), to_date(end)

        def parse(inst, rows, s, e):  # noqa: ANN001,ANN202
            bars = self.parse_prices(inst, rows, s, e)
            return None if bars.empty else bars

        frames, errors = self._batch(instruments, start_d, end_d, parse)
        actions = [
            self.parse_actions(inst, rows) for inst, rows in []
        ]  # actions come via get_corporate_actions
        del actions
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_bars(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": start_d.isoformat(), "end": end_d.isoformat(), "adjusted": False},
        )

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        frames, errors = self._batch(instruments, start_d, end_d, lambda i, r, s, e: self.parse_actions(i, r))
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_actions(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": start_d.isoformat(), "end": end_d.isoformat()},
        )

    def parse_prices(
        self, inst: Instrument, rows: Sequence[dict[str, Any]], start: date, end: date
    ) -> pd.DataFrame:
        if not rows:
            return empty_bars()
        fm = self.field_map
        frame = pd.DataFrame(
            {
                "session_date": pd.to_datetime([r["date"] for r in rows], utc=True)
                .tz_localize(None)
                .normalize(),
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

    def parse_actions(self, inst: Instrument, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
        out: list[tuple[str, object, str, float, str | None]] = []
        fm = self.field_map
        for r in rows:
            d = (
                pd.Timestamp(r["date"]).tz_localize(None)
                if pd.Timestamp(r["date"]).tzinfo is None
                else pd.Timestamp(r["date"]).tz_convert(None)
            )
            div = float(r.get(fm.dividends) or 0.0) if fm.dividends else 0.0
            spl = float(r.get(fm.splits) or 1.0) if fm.splits else 1.0
            if div:
                out.append(
                    (inst.instrument_id, d, "dividend", div * float(inst.price_scale), inst.currency.value)
                )
            if spl and spl != 1.0:
                out.append((inst.instrument_id, d, "split", spl, None))
        return actions_frame(out, self.name)
