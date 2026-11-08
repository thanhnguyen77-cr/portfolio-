"""Ken French data library loader (US factors; the UK has no free equivalent - build UK factors from the
universe instead). Files are zipped CSVs with preamble lines; values are in percent."""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from quant.data.adapters.base import BaseHTTPAdapter, Capability, RateLimit

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff3_daily": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5_daily": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "momentum_daily": "F-F_Momentum_Factor_daily_CSV.zip",
}


class KenFrenchSource(BaseHTTPAdapter):
    name = "ken_french"
    rate_limit = RateLimit(min_interval_s=1.0)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.MACRO})

    def fetch_zip(self, key: str) -> bytes:
        if key not in FILES:
            raise ValueError(f"unknown Ken French file {key!r}; choose from {sorted(FILES)}")
        return self._http("GET", BASE + FILES[key]).content


def parse_ken_french_zip(payload: bytes) -> pd.DataFrame:
    """First CSV in the zip -> DataFrame indexed by date with decimal (not percent) values."""
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(name).decode("latin-1")
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip()[:8].isdigit() and len(line.strip()[:8]) == 8
    )
    header_idx = start - 1
    header = [h.strip() for h in lines[header_idx].split(",")]
    header[0] = "date"
    rows = []
    for line in lines[start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break  # the daily block ends where the annual/monthly block (or a blank line) begins
        rows.append(parts)
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")
    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.columns = [c.replace("-", "_").lower() for c in df.columns]
    return df


def load_ken_french(
    key: str = "ff5_daily", source: KenFrenchSource | None = None, payload: bytes | None = None
) -> pd.DataFrame:
    """Daily factor returns as decimals (mkt_rf, smb, hml, rmw, cma, rf; or mom)."""
    data = payload if payload is not None else (source or KenFrenchSource()).fetch_zip(key)
    return parse_ken_french_zip(data)
