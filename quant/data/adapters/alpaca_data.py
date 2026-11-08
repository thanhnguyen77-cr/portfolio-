"""Alpaca Market Data v2 daily bars (US equities; IEX feed on the free tier).

Needs QUANT_ALPACA_API_KEY / QUANT_ALPACA_SECRET_KEY. Requests are batched by symbol list and paginated
with `next_page_token`. `adjustment=raw` is requested so stored bars are `raw` and the engine applies
corporate actions itself. Timestamps are session-start UTC (e.g. 05:00Z for a US session); the session
date is taken in New York time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.time import to_date
from quant.core.types import AdjustmentState, Instrument, Market
from quant.data.adapters._bars import canonical_bars
from quant.data.adapters.base import (
    AdapterResult,
    BaseHTTPAdapter,
    Capability,
    FieldMap,
    RateLimit,
    empty_bars,
)

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
US_MARKETS = {Market.XNYS.value, Market.XNAS.value}


class AlpacaDataAdapter(BaseHTTPAdapter):
    name = "alpaca_data"
    identifier_key = "alpaca"
    rate_limit = RateLimit(requests_per_minute=200, min_interval_s=0.3, note="free tier: 200 req/min")
    field_map = FieldMap(
        open="o", high="h", low="l", close="c", volume="v", adj_close=None, dividends=None, splits=None
    )
    adjustment_state = AdjustmentState.RAW

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        *,
        feed: str = "iex",
        session=None,  # noqa: ANN001
        drop_unfinished_session: bool = True,
    ) -> None:
        super().__init__(session=session)
        self.api_key, self.secret_key, self.feed = api_key, secret_key, feed
        self.drop_unfinished_session = drop_unfinished_session

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BARS_DAILY})

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise AdapterNotConfigured(
                "Alpaca market data needs QUANT_ALPACA_API_KEY and QUANT_ALPACA_SECRET_KEY"
            )
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        if freq != "1d":
            raise AdapterError(f"{self.name}: only daily bars are supported (got {freq!r})")
        start_d, end_d = to_date(start), to_date(end)
        headers = self._headers()
        errors: dict[str, str] = {}
        by_symbol: dict[str, Instrument] = {}
        for inst in instruments:
            if inst.market.value not in US_MARKETS:
                errors[inst.instrument_id] = f"Alpaca serves US equities only ({inst.market.value})"
                continue
            by_symbol[self.symbol_for(inst)] = inst
        payload: dict[str, list[dict[str, Any]]] = {s: [] for s in by_symbol}
        if by_symbol:
            params: dict[str, Any] = {
                "symbols": ",".join(sorted(by_symbol)),
                "timeframe": "1Day",
                "start": start_d.isoformat(),
                "end": (end_d + timedelta(days=1)).isoformat(),
                "adjustment": "raw",
                "feed": self.feed,
                "limit": 10000,
            }
            try:
                token = None
                for _ in range(200):  # pagination guard
                    if token:
                        params["page_token"] = token
                    body = self._http("GET", BARS_URL, params=params, headers=headers).json()
                    for sym, rows in (body.get("bars") or {}).items():
                        payload.setdefault(sym, []).extend(rows or [])
                    token = body.get("next_page_token")
                    if not token:
                        break
            except Exception as exc:  # noqa: BLE001
                for inst in by_symbol.values():
                    errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                payload = {}
        frames = []
        for sym, rows in payload.items():
            inst = by_symbol[sym]
            bars = self.parse_rows(inst, rows, start_d, end_d)
            if bars.empty:
                errors[inst.instrument_id] = f"no bars for {sym} in {start_d}..{end_d}"
                continue
            frames.append(bars)
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_bars(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "adjustment": "raw",
                "feed": self.feed,
            },
        )

    def parse_rows(
        self, inst: Instrument, rows: Sequence[Mapping[str, Any]], start: date, end: date
    ) -> pd.DataFrame:
        if not rows:
            return empty_bars()
        fm = self.field_map
        ts = pd.to_datetime([r["t"] for r in rows], utc=True).tz_convert("America/New_York")
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
