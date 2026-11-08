"""Companies House (UK) public data API: company profile and filing history (accounts, confirmation
statements, ...). Needs QUANT_COMPANIES_HOUSE_API_KEY (HTTP basic auth, key as the username).

Instruments map through `identifiers["companies_house"]` (the 8-character company number). This is a
filings index with true filing dates - the UK counterpart of EDGAR submissions - not a financial-statement
parser: UK accounts arrive as iXBRL documents that are out of scope here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.core.time import now, to_date
from quant.core.types import Instrument
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit

BASE_URL = "https://api.company-information.service.gov.uk"
FILING_COLUMNS = (
    "instrument_id",
    "filed_at",
    "period_end",
    "form",
    "category",
    "description",
    "accession",
    "url",
    "vendor",
    "recorded_at",
)


def empty_filings() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FILING_COLUMNS})


class CompaniesHouseAdapter(BaseHTTPAdapter):
    name = "companies_house"
    identifier_key = "companies_house"
    rate_limit = RateLimit(min_interval_s=0.11, requests_per_minute=600, note="600 requests per 5 minutes")

    def __init__(self, api_key: str | None = None, session=None) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.api_key = api_key

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.FILINGS})

    def _auth(self) -> tuple[str, str]:
        if not self.api_key:
            raise AdapterNotConfigured("Companies House needs QUANT_COMPANIES_HOUSE_API_KEY")
        return (self.api_key, "")

    def company_number(self, inst: Instrument) -> str:
        number = inst.identifiers.get("companies_house")
        if not number:
            raise AdapterError(f"{inst.instrument_id} has no identifiers['companies_house'] company number")
        return str(number).upper().zfill(8)

    def get_profile(self, inst: Instrument) -> dict[str, Any]:
        return self._http("GET", f"{BASE_URL}/company/{self.company_number(inst)}", auth=self._auth()).json()

    def get_filings(
        self,
        instruments: Sequence[Instrument],
        *,
        categories: Sequence[str] = ("accounts", "confirmation-statement"),
        since: date | None = None,
        items_per_page: int = 100,
    ) -> AdapterResult:
        self._auth()  # credentials are checked once, loudly, not swallowed per instrument
        frames, errors = [], {}
        for inst in instruments:
            try:
                number = self.company_number(inst)
                items: list[dict[str, Any]] = []
                start_index = 0
                for _ in range(50):
                    params = {"items_per_page": items_per_page, "start_index": start_index}
                    if categories:
                        params["category"] = ",".join(categories)
                    body = self._http(
                        "GET", f"{BASE_URL}/company/{number}/filing-history", params=params, auth=self._auth()
                    ).json()
                    page = body.get("items") or []
                    items.extend(page)
                    start_index += len(page)
                    if not page or start_index >= int(body.get("total_count") or 0):
                        break
                frame = self.parse_filings(inst, items, since)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if frame.empty:
                errors[inst.instrument_id] = "no filings in the requested categories"
                continue
            frames.append(frame)
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else empty_filings(),
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"categories": list(categories), "since": since.isoformat() if since else None},
        )

    def parse_filings(
        self, inst: Instrument, items: Sequence[dict[str, Any]], since: date | None
    ) -> pd.DataFrame:
        rows = []
        for it in items:
            filed = it.get("date")
            if not filed:
                continue
            filed_d = to_date(filed)
            if since and filed_d < since:
                continue
            desc_values = it.get("description_values") or {}
            period_end = desc_values.get("made_up_date") or desc_values.get("period_end_on")
            rows.append(
                {
                    "instrument_id": inst.instrument_id,
                    "filed_at": pd.Timestamp(filed_d),
                    "period_end": pd.Timestamp(to_date(period_end)) if period_end else pd.NaT,
                    "form": it.get("type"),
                    "category": it.get("category"),
                    "description": it.get("description"),
                    "accession": it.get("transaction_id"),
                    "url": (it.get("links") or {}).get("document_metadata")
                    or (it.get("links") or {}).get("self"),
                    "vendor": self.name,
                    "recorded_at": pd.Timestamp(now()),
                }
            )
        if not rows:
            return empty_filings()
        return pd.DataFrame(rows, columns=FILING_COLUMNS).sort_values("filed_at").reset_index(drop=True)
