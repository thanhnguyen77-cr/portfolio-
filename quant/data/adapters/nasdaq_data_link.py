"""Nasdaq Data Link (formerly Quandl) time-series datasets as macro series.

Series ids are `DATABASE/DATASET` with an optional `.column` suffix (default: the first numeric column),
e.g. `ML/EMHYY` or `LBMA/GOLD.2`. An API key (QUANT_NDL_API_KEY) lifts the anonymous quota and unlocks
premium databases; without one, free datasets still work within the daily limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from quant.core.errors import AdapterError
from quant.core.time import to_date
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit
from quant.data.adapters.fred import _empty_macro, macro_frame

DATA_URL = "https://data.nasdaq.com/api/v3/datasets/{database}/{dataset}/data.json"


def split_series_id(series_id: str) -> tuple[str, str, str | None]:
    """'LBMA/GOLD.2' -> ('LBMA', 'GOLD', '2'); 'ML/EMHYY' -> ('ML', 'EMHYY', None)."""
    if "/" not in series_id:
        raise AdapterError(f"NDL series id must be DATABASE/DATASET[.column], got {series_id!r}")
    database, rest = series_id.split("/", 1)
    column = None
    if "." in rest:
        rest, column = rest.split(".", 1)
    return database.upper(), rest.upper(), column


class NasdaqDataLinkAdapter(BaseHTTPAdapter):
    name = "ndl"
    identifier_key = "ndl"
    rate_limit = RateLimit(min_interval_s=0.5, note="anonymous: 50 calls/day; keyed: 300 calls/10s")

    def __init__(self, api_key: str | None = None, session=None) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.api_key = api_key

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.MACRO})

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        frames, errors = [], {}
        for sid in series_ids:
            try:
                database, dataset, column = split_series_id(sid)
                params: dict[str, Any] = {
                    "start_date": start_d.isoformat(),
                    "end_date": end_d.isoformat(),
                    "order": "asc",
                }
                if self.api_key:
                    params["api_key"] = self.api_key
                body = self._http(
                    "GET", DATA_URL.format(database=database, dataset=dataset), params=params
                ).json()
                frame = self.parse_payload(sid, body, column)
            except Exception as exc:  # noqa: BLE001
                errors[sid] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if frame.empty:
                errors[sid] = f"no observations for {sid} in {start_d}..{end_d}"
                continue
            frames.append(frame)
        return AdapterResult(
            vendor=self.name,
            frame=pd.concat(frames, ignore_index=True) if frames else _empty_macro(),
            errors=errors,
            requested=list(series_ids),
            params={"start": start_d.isoformat(), "end": end_d.isoformat()},
        )

    def parse_payload(self, series_id: str, body: dict[str, Any], column: str | None) -> pd.DataFrame:
        data = (body or {}).get("dataset_data") or {}
        cols = [str(c) for c in data.get("column_names") or []]
        rows = data.get("data") or []
        if not cols or not rows:
            return _empty_macro()
        df = pd.DataFrame(rows, columns=cols)
        date_col = cols[0]
        if column is None:
            numeric = [c for c in cols[1:] if pd.to_numeric(df[c], errors="coerce").notna().any()]
            if not numeric:
                raise AdapterError(f"{series_id}: no numeric column in {cols}")
            value_col = numeric[0]
        elif column.isdigit():
            idx = int(column)
            if idx < 1 or idx >= len(cols):
                raise AdapterError(f"{series_id}: column index {idx} out of range for {cols}")
            value_col = cols[idx]
        else:
            matches = [c for c in cols if c.lower() == column.lower()]
            if not matches:
                raise AdapterError(f"{series_id}: column {column!r} not in {cols}")
            value_col = matches[0]
        return macro_frame(series_id, df[date_col], df[value_col], self.name)
