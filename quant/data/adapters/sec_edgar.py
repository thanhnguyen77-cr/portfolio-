"""SEC EDGAR XBRL company facts: fundamentals with true filing dates (US filers only).

The SEC requires a User-Agent that identifies the caller with contact details (`QUANT_SEC_USER_AGENT`,
e.g. "Jane Doe jane@example.com") and asks for at most 10 requests per second. Tickers map to CIKs through
the public `company_tickers.json`; the catalogue's `identifiers["cik"]` takes precedence when present.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.time import now, to_date
from quant.core.types import Instrument
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"


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
        return frozenset({Capability.FUNDAMENTALS, Capability.FILINGS})

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

    # ------------------------------------------------------------------ filings index
    def get_filings(
        self,
        instruments: Sequence[Instrument],
        *,
        forms: Sequence[str] = ("10-K", "10-Q", "8-K", "20-F", "6-K"),
        since: date | None = None,
    ) -> AdapterResult:
        """Recent filings per company from the submissions API (true filing dates, document URLs)."""
        self._require_config()
        frames, errors = [], {}
        for inst in instruments:
            try:
                cik = self.cik_for(inst)
                payload = self._http("GET", SUBMISSIONS_URL.format(cik=cik)).json()
                frame = self.parse_submissions(inst, cik, payload, forms, since)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if frame.empty:
                errors[inst.instrument_id] = f"no filings of forms {list(forms)}"
                continue
            frames.append(frame)
        from quant.data.adapters.companies_house import empty_filings

        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_filings(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"forms": list(forms), "since": since.isoformat() if since else None},
        )

    @staticmethod
    def parse_submissions(
        inst: Instrument, cik: int, payload: Mapping[str, Any], forms: Sequence[str], since: date | None
    ) -> pd.DataFrame:
        from quant.data.adapters.companies_house import FILING_COLUMNS, empty_filings

        recent = (payload.get("filings") or {}).get("recent") or {}
        n = len(recent.get("accessionNumber") or [])
        wanted = {f.upper() for f in forms} if forms else None
        rows = []
        for i in range(n):
            form = str(recent["form"][i])
            if wanted and form.upper() not in wanted:
                continue
            filed = to_date(recent["filingDate"][i])
            if since and filed < since:
                continue
            accession = str(recent["accessionNumber"][i])
            doc = (recent.get("primaryDocument") or [None] * n)[i]
            report = (recent.get("reportDate") or [None] * n)[i]
            rows.append(
                {
                    "instrument_id": inst.instrument_id,
                    "filed_at": pd.Timestamp(filed),
                    "period_end": pd.Timestamp(to_date(report)) if report else pd.NaT,
                    "form": form,
                    "category": "financial" if form.upper() in {"10-K", "10-Q", "20-F", "6-K"} else "event",
                    "description": (recent.get("primaryDocDescription") or [None] * n)[i],
                    "accession": accession,
                    "url": ARCHIVE_URL.format(
                        cik=cik, accession_nodash=accession.replace("-", ""), document=doc
                    )
                    if doc
                    else None,
                    "vendor": "sec_edgar",
                    "recorded_at": pd.Timestamp(now()),
                }
            )
        if not rows:
            return empty_filings()
        return pd.DataFrame(rows, columns=FILING_COLUMNS).sort_values("filed_at").reset_index(drop=True)
