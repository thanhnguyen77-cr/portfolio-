"""SEC EDGAR XBRL company facts: fundamentals with true filing dates (US filers only).

The SEC requires a User-Agent that identifies the caller with contact details (`QUANT_SEC_USER_AGENT`,
e.g. "Jane Doe jane@example.com") and asks for at most 10 requests per second. Tickers map to CIKs through
the public `company_tickers.json`; the catalogue's `identifiers["cik"]` takes precedence when present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.types import Instrument
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


class SECEdgarAdapter(BaseHTTPAdapter):
    name = "sec_edgar"
    identifier_key = "cik"
    rate_limit = RateLimit(
        min_interval_s=0.12, requests_per_minute=500, note="SEC fair-access: <= 10 requests/s"
    )

    def __init__(self, user_agent: str | None = None, session=None) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.contact_user_agent = user_agent
        if user_agent:
            self.user_agent = user_agent
        self._tickers: dict[str, int] | None = None

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.FUNDAMENTALS})

    def _require_config(self) -> None:
        if not self.contact_user_agent:
            raise AdapterNotConfigured(
                "SEC EDGAR requires a User-Agent with contact details: "
                "set QUANT_SEC_USER_AGENT='Name email@example.com'"
            )

    @staticmethod
    def parse_tickers(payload: Mapping[str, Any]) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in payload.values():
            ticker = str(entry.get("ticker", "")).upper().replace(".", "-")
            if ticker:
                out[ticker] = int(entry["cik_str"])
        return out

    def cik_for(self, inst: Instrument) -> int:
        if "cik" in inst.identifiers:
            return int(inst.identifiers["cik"])
        if self._tickers is None:
            self._tickers = self.parse_tickers(self._http("GET", TICKERS_URL).json())
        symbol = inst.identifiers.get("yahoo", inst.symbol).upper()
        if symbol not in self._tickers:
            raise AdapterError(f"no CIK found for {inst.instrument_id} ({symbol})")
        return self._tickers[symbol]

    def get_fundamentals(self, instruments: Sequence[Instrument]) -> AdapterResult:
        from quant.fundamental.statements import from_edgar_companyfacts

        self._require_config()
        frames, errors = [], {}
        for inst in instruments:
            try:
                cik = self.cik_for(inst)
                facts = self._http("GET", FACTS_URL.format(cik=cik)).json()
                rows = from_edgar_companyfacts(inst, facts)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if rows.empty:
                errors[inst.instrument_id] = "no us-gaap facts mapped"
                continue
            frames.append(rows)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"source": "companyfacts; true filing dates"},
        )
