"""Office for National Statistics time series (UK) from the ONS website's JSON endpoint.

The dedicated `api.ons.gov.uk/timeseries` API was retired on 25 November 2024 (its 404 body says so). The
same series are served by the website as `https://www.ons.gov.uk/<taxonomy>/timeseries/<series>/<dataset>/data`,
where <taxonomy> is the section the dataset lives under. Series are addressed as `DATASET/SERIES` (e.g.
`MM23/L55O` for CPIH annual rate) for the datasets in `DATASET_PATHS`, or as a full site URI such as
`/economy/inflationandpriceindices/timeseries/l55o/mm23` for anything else. Monthly observations are
preferred, then quarterly, then annual; dates are the start of the period. ONS revises history, so
re-fetching can restate values (the store keeps the latest).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import pandas as pd

from quant.core.time import to_date
from quant.data.adapters.base import AdapterResult, BaseHTTPAdapter, Capability, RateLimit
from quant.data.adapters.fred import _empty_macro, macro_frame

ONS_BASE = "https://www.ons.gov.uk"
# dataset id -> website section (verified live on 2026-11-08); extend or pass a full URI as the series id
DATASET_PATHS: dict[str, str] = {
    "MM23": "/economy/inflationandpriceindices",
    "LMS": "/employmentandlabourmarket/peoplenotinwork/unemployment",  # MGSX; other LMS series need a URI
    "MGDP": "/economy/grossdomesticproductgdp",
    "DRSI": "/businessindustryandtrade/retailindustry",
}
_MONTHS = {
    m: i
    for i, m in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
    )
}


def parse_ons_date(label: str) -> pd.Timestamp | None:
    s = str(label).strip().upper()
    m = re.fullmatch(r"(\d{4}) ([A-Z]{3})", s)
    if m and m.group(2) in _MONTHS:
        return pd.Timestamp(int(m.group(1)), _MONTHS[m.group(2)], 1)
    q = re.fullmatch(r"(\d{4}) Q([1-4])", s)
    if q:
        return pd.Timestamp(int(q.group(1)), 3 * (int(q.group(2)) - 1) + 1, 1)
    if re.fullmatch(r"\d{4}", s):
        return pd.Timestamp(int(s), 1, 1)
    return None


class ONSAdapter(BaseHTTPAdapter):
    name = "ons"
    identifier_key = "ons"
    rate_limit = RateLimit(min_interval_s=0.5)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.MACRO})

    @staticmethod
    def split_id(series_id: str) -> tuple[str, str]:
        """'MM23/L55O' -> ('MM23', 'L55O'); a full URI '/.../timeseries/l55o/mm23' -> ('MM23', 'L55O')."""
        sid = series_id.strip()
        if sid.startswith("/"):
            parts = [p for p in sid.split("/") if p]
            if len(parts) < 3 or parts[-3].lower() != "timeseries":
                raise ValueError(f"ONS URI must end in /timeseries/<series>/<dataset>, got {series_id!r}")
            return parts[-1].upper(), parts[-2].upper()
        if "/" not in sid:
            raise ValueError(f"ONS series must be DATASET/SERIES or a site URI, got {series_id!r}")
        dataset, series = sid.split("/", 1)
        return dataset.upper(), series.upper()

    @staticmethod
    def canonical_id(series_id: str) -> str:
        """'mm23/l55o' -> 'MM23/L55O'; a site URI is kept as a lower-case path."""
        sid = series_id.strip()
        if sid.startswith("/"):
            ONSAdapter.split_id(sid)  # validate
            return sid.rstrip("/").lower()
        dataset, series = ONSAdapter.split_id(sid)
        return f"{dataset}/{series}"

    @staticmethod
    def url_for(series_id: str) -> str:
        """Website JSON URL for a `DATASET/SERIES` id (known datasets) or a full site URI."""
        sid = series_id.strip()
        if sid.startswith("/"):
            ONSAdapter.split_id(sid)  # validate
            return f"{ONS_BASE}{sid.rstrip('/').lower()}/data"
        dataset, series = ONSAdapter.split_id(sid)
        section = DATASET_PATHS.get(dataset)
        if section is None:
            raise ValueError(
                f"no website section known for ONS dataset {dataset!r}; pass the full URI "
                f"(e.g. '/economy/inflationandpriceindices/timeseries/{series.lower()}/{dataset.lower()}') "
                f"or add it to DATASET_PATHS"
            )
        return f"{ONS_BASE}{section}/timeseries/{series.lower()}/{dataset.lower()}/data"

    @staticmethod
    def parse_payload(series_id: str, payload: Mapping[str, Any], vendor: str = "ons") -> pd.DataFrame:
        for key in ("months", "quarters", "years"):
            obs = payload.get(key) or []
            if obs:
                dates = [parse_ons_date(o.get("date", "")) for o in obs]
                values = [o.get("value") for o in obs]
                keep = [i for i, d in enumerate(dates) if d is not None]
                if not keep:
                    continue
                return macro_frame(
                    series_id,
                    pd.Series([dates[i] for i in keep]),
                    pd.Series([values[i] for i in keep]),
                    vendor,
                )
        return _empty_macro()

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        start_d, end_d = to_date(start), to_date(end)
        frames, errors = [], {}
        for sid in series_ids:
            try:
                payload = self._http("GET", self.url_for(sid)).json()
                df = self.parse_payload(self.canonical_id(sid), payload, self.name)
                df = df[(df["date"] >= pd.Timestamp(start_d)) & (df["date"] <= pd.Timestamp(end_d))]
            except Exception as exc:  # noqa: BLE001
                errors[sid] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if df.empty:
                errors[sid] = f"no observations for {sid} in {start_d}..{end_d}"
                continue
            frames.append(df.reset_index(drop=True))
        frame = pd.concat(frames, ignore_index=True) if frames else _empty_macro()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=list(series_ids),
            params={"start": start_d.isoformat(), "end": end_d.isoformat()},
        )
