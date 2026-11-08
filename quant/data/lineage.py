"""Data lineage: every fetch, transform and write leaves a row saying what happened, from where, to what."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from typing import TYPE_CHECKING, Any

import pandas as pd

from quant import __version__
from quant.core.time import now

if TYPE_CHECKING:
    from quant.data.store import DuckDBStore

_TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "pyarrow",
    "duckdb",
    "yfinance",
    "exchange-calendars",
    "statsmodels",
    "arch",
)


def package_versions() -> dict[str, str]:
    """Versions of quant and the libraries that shape the data, for reproducibility."""
    out = {"quant": __version__}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def frame_digest(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 of a frame's contents (order-sensitive, index ignored)."""
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    hashed = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


@dataclass
class LineageRecord:
    vendor: str
    operation: str  # fetch | transform | write | validate
    instrument_ids: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    rows: int = 0
    sha256: str | None = None
    diff: dict[str, Any] = field(default_factory=dict)
    note: str | None = None
    lineage_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime = field(default_factory=now)
    versions: dict[str, str] = field(default_factory=package_versions)

    def to_row(self) -> tuple[Any, ...]:
        return (
            self.lineage_id,
            self.ts,
            self.vendor,
            self.operation,
            json.dumps(sorted(self.instrument_ids)),
            json.dumps(self.params, default=str, sort_keys=True),
            int(self.rows),
            self.sha256,
            json.dumps(self.versions, sort_keys=True),
            json.dumps(self.diff, default=str, sort_keys=True),
            self.note,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LineageLog:
    """Writes `LineageRecord`s to the DuckDB `lineage` table and reads them back."""

    def __init__(self, store: DuckDBStore) -> None:
        self._store = store

    def record(self, rec: LineageRecord) -> LineageRecord:
        with self._store.write() as con:
            con.execute(
                "INSERT INTO lineage (lineage_id, ts, vendor, operation, instrument_ids, params, rows, "
                "sha256, versions, diff, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                list(rec.to_row()),
            )
        return rec

    def recent(self, limit: int = 50) -> pd.DataFrame:
        return self._store.query("SELECT * FROM lineage ORDER BY ts DESC LIMIT ?", [int(limit)])

    def for_instrument(self, instrument_id: str) -> pd.DataFrame:
        needle = f'%"{instrument_id}"%'
        return self._store.query(
            "SELECT * FROM lineage WHERE instrument_ids LIKE ? ORDER BY ts DESC", [needle]
        )
