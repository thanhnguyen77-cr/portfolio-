"""FRED (Federal Reserve Bank of St. Louis) macro series.

Without an API key the public CSV endpoint is used (one request per series). With `QUANT_FRED_API_KEY`
the JSON observations API is used instead, which respects date bounds server-side. FRED marks missing
observations with "."; those rows are dropped rather than stored as zeros.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import date

import pandas as pd

from quant.core.time import now, to_date
from quant.data.adapters.base import (
    MACRO_COLUMNS,
    AdapterResult,
    BaseHTTPAdapter,
    Capability,
    RateLimit,
)

CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
API_URL = "https://api.stlouisfed.org/fred/series/observations"


def _empty_macro() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MACRO_COLUMNS})


def macro_frame(series_id: str, dates: pd.Series, values: pd.Series, vendor: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "series_id": series_id,
            "date": pd.to_datetime(dates).dt.normalize().astype("datetime64[ns]"),
            "value": pd.to_numeric(values, errors="coerce").astype("float64"),
        }
    ).dropna(subset=["value"])
    df["vendor"] = vendor
    df["ingested_at"] = pd.Timestamp(now())
    return df.loc[:, list(MACRO_COLUMNS)].reset_index(drop=True)


class FREDAdapter(BaseHTTPAdapter):
    name = "fred"
    identifier_key = "fred"
    rate_limit = RateLimit(min_interval_s=0.5, requests_per_minute=100)

    def __init__(self, api_key: str | None = None, session=None) -> None:  # noqa: ANN001
        super().__init__(session=session)
        self.api_key = api_key

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.MACRO})

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        frames: list[pd.DataFrame] = []
        errors: dict[str, str] = {}
        for sid in series_ids:
            try:
                df = self._fetch_one(sid, start_d, end_d)
            except Exception as exc:  # noqa: BLE001 - per-series failure is reported, not raised
                errors[sid] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if df.empty:
                errors[sid] = f"no observations for {sid} in {start_d}..{end_d}"
                continue
            frames.append(df)
        frame = pd.concat(frames, ignore_index=True) if frames else _empty_macro()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=list(series_ids),
            params={"start": start_d.isoformat(), "end": end_d.isoformat(), "api": bool(self.api_key)},
        )

    def _fetch_one(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        if self.api_key:
            resp = self._http(
                "GET",
                API_URL,
                params={
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": start.isoformat(),
                    "observation_end": end.isoformat(),
                },
            )
            obs = pd.DataFrame(resp.json().get("observations", []))
            if obs.empty:
                return _empty_macro()
            return macro_frame(series_id, obs["date"], obs["value"].replace(".", None), self.name)
        resp = self._http("GET", CSV_URL, params={"id": series_id})
        return self.parse_csv(series_id, resp.text, start, end)

    def parse_csv(self, series_id: str, text: str, start: date, end: date) -> pd.DataFrame:
        """Parse the fredgraph CSV (header `observation_date,<ID>` or legacy `DATE,<ID>`; '.' = missing)."""
        raw = pd.read_csv(io.StringIO(text))
        if raw.shape[1] < 2:
            raise ValueError(f"unexpected FRED CSV shape {raw.shape} for {series_id}")
        date_col, value_col = raw.columns[0], raw.columns[1]
        values = raw[value_col].replace(".", None)
        df = macro_frame(series_id, raw[date_col], values, self.name)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df[mask].reset_index(drop=True)
