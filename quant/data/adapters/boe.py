"""Bank of England Statistical Interactive Database (IADB) CSV export.

One request fetches several series codes (e.g. IUDBEDR Bank Rate, IUDSOIA SONIA). The export returns
`DATE,<CODE>,<CODE>...` with dates like `02 Jan 2024`; a requested code missing from the response is
reported as a per-series error.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import date

import pandas as pd

from quant.core.time import to_date
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit
from quant.data.adapters.fred import _empty_macro, macro_frame

IADB_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"


class BankOfEnglandAdapter(BaseHTTPAdapter):
    name = "boe"
    identifier_key = "boe"
    rate_limit = RateLimit(min_interval_s=1.0, note="public website export; be gentle")

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.MACRO})

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        ids = [s.upper() for s in series_ids]
        params = {
            "csv.x": "yes",
            "Datefrom": start_d.strftime("%d/%b/%Y"),
            "Dateto": end_d.strftime("%d/%b/%Y"),
            "SeriesCodes": ",".join(ids),
            "CSVF": "TN",
            "UsingCodes": "Y",
            "VPD": "Y",
            "VFD": "N",
        }
        errors: dict[str, str] = {}
        try:
            resp = self._http("GET", IADB_URL, params=params)
            frame = self.parse_csv(resp.text, ids, errors)
        except Exception as exc:  # noqa: BLE001 - the whole batch failed; report every series
            frame = _empty_macro()
            for sid in ids:
                errors[sid] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=ids,
            params={"start": start_d.isoformat(), "end": end_d.isoformat()},
        )

    def parse_csv(self, text: str, requested: Sequence[str], errors: dict[str, str]) -> pd.DataFrame:
        raw = pd.read_csv(io.StringIO(text))
        if raw.empty or raw.columns[0].strip().upper() != "DATE":
            raise ValueError("unexpected IADB CSV: first column must be DATE")
        raw.columns = [c.strip().upper() for c in raw.columns]
        dates = pd.to_datetime(raw["DATE"], format="%d %b %Y", errors="coerce")
        if dates.isna().all():
            dates = pd.to_datetime(raw["DATE"], errors="coerce")
        frames = []
        for sid in requested:
            if sid not in raw.columns:
                errors[sid] = f"series {sid} not in response (columns: {list(raw.columns)})"
                continue
            df = macro_frame(sid, dates, raw[sid], self.name)
            if df.empty:
                errors[sid] = f"no observations for {sid}"
                continue
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else _empty_macro()
