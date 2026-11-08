"""Local storage: Parquet files for bar series, DuckDB for relational tables and SQL across both.

Layout: `<parquet_dir>/vendor=<vendor>/freq=<freq>/<instrument_id>.parquet`, one file per series, hive-style
so DuckDB exposes `vendor` and `freq` as columns. Appends are read -> dedupe -> rewrite with an atomic
rename; the key is (instrument_id, session_date) and the row with the latest `ingested_at` wins, so
re-ingesting the same data is a no-op and a vendor restatement replaces the old row.

Prices are stored exactly as the vendor delivered them (see `AdjustmentState`); adjustment happens at read
time from the corporate_actions table, never in storage.

DuckDB is single-writer: readers use `read()` (read-only connections), writers use `write()` under a file
lock. A different process may hold the write lock; readers are never blocked by it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from quant.core.errors import StoreError
from quant.core.time import now, to_date
from quant.core.types import BAR_COLUMNS, Instrument, bar_schema

BAR_KEY: tuple[str, str] = ("instrument_id", "session_date")
_STRING_COLS = ("instrument_id", "currency", "vendor", "adjustment_state")
_FLOAT_COLS = ("open", "high", "low", "close", "volume")
_TS_COLS = ("ts_close_utc", "ingested_at")
_PRICE_COLS = ("open", "high", "low", "close")


# ----------------------------------------------------------------------------- frame normalisation


def coerce_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a bars frame to the canonical columns and dtypes (pandas 2 and 3 safe).

    session_date -> tz-naive datetime64[ns] at midnight; timestamps -> datetime64[ns, UTC]; prices float64;
    strings -> plain str. Sorted by (instrument_id, session_date). Raises StoreError on missing columns,
    nulls in key/string columns, or rows without a close.
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise StoreError(f"bars frame is missing columns {missing}")
    out = df.loc[:, list(BAR_COLUMNS)].copy()
    if out.empty:
        return _empty_bars()
    for col in _STRING_COLS:
        if out[col].isna().any():
            raise StoreError(f"column {col!r} contains nulls")
        out[col] = out[col].astype(str)
    for col in _FLOAT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    if out["close"].isna().any():
        raise StoreError("bars frame has rows without a close price")
    sd = pd.to_datetime(out["session_date"])
    if sd.dt.tz is not None:
        raise StoreError("session_date must be a tz-naive calendar date")
    out["session_date"] = sd.dt.normalize().astype("datetime64[ns]")
    for col in _TS_COLS:
        out[col] = pd.to_datetime(out[col], utc=True).astype("datetime64[ns, UTC]")
    out = out.sort_values([*BAR_KEY, "ingested_at"], kind="mergesort").reset_index(drop=True)
    return out


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "action_type": pd.Series(dtype=str),
            "value": pd.Series(dtype="float64"),
            "currency": pd.Series(dtype=object),
            "vendor": pd.Series(dtype=str),
            "recorded_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instrument_id": pd.Series(dtype=str),
            "session_date": pd.Series(dtype="datetime64[ns]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "currency": pd.Series(dtype=str),
            "vendor": pd.Series(dtype=str),
            "adjustment_state": pd.Series(dtype=str),
            "ts_close_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "ingested_at": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _to_arrow(bars: pd.DataFrame) -> pa.Table:
    schema = bar_schema()
    arrays: list[pa.Array] = []
    for fld in schema:
        col = bars[fld.name]
        if fld.name == "session_date":
            arrays.append(pa.array(col.dt.date.tolist(), type=pa.date32()))
        elif fld.name in _TS_COLS:
            arrays.append(pa.array(col, type=fld.type))
        elif fld.name in _FLOAT_COLS:
            arrays.append(pa.array(col.to_numpy(dtype="float64"), type=pa.float64(), from_pandas=True))
        else:
            arrays.append(pa.array(col.tolist(), type=pa.string()))
    return pa.Table.from_arrays(arrays, schema=schema)


def _from_arrow(table: pa.Table) -> pd.DataFrame:
    df = table.to_pandas(date_as_object=False)
    return coerce_bars(df)


def dedupe_bars(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Union on the bar key; the row with the latest `ingested_at` wins."""
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.sort_values([*BAR_KEY, "ingested_at"], kind="mergesort")
    deduped = combined.drop_duplicates(list(BAR_KEY), keep="last")
    return deduped.reset_index(drop=True)


# ----------------------------------------------------------------------------- Parquet store


@dataclass(frozen=True)
class WriteResult:
    instrument_id: str
    path: Path
    rows_before: int
    rows_after: int
    rows_new: int
    rows_updated: int
    rows_unchanged: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "instrument_id": self.instrument_id,
            "path": str(self.path),
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_new": self.rows_new,
            "rows_updated": self.rows_updated,
            "rows_unchanged": self.rows_unchanged,
        }


def _diff_counts(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[int, int, int]:
    """(new, updated, unchanged) keys in `incoming` relative to `existing`, comparing OHLCV values."""
    if existing.empty:
        return len(incoming), 0, 0
    merged = incoming.merge(existing, on=list(BAR_KEY), how="left", suffixes=("", "_old"), indicator=True)
    is_new = merged["_merge"] == "left_only"
    old_vals = merged[[f"{c}_old" for c in _FLOAT_COLS]].to_numpy(dtype="float64")
    new_vals = merged[list(_FLOAT_COLS)].to_numpy(dtype="float64")
    same = np.all(np.isclose(old_vals, new_vals, rtol=1e-9, atol=1e-12, equal_nan=True), axis=1)
    unchanged = int((~is_new & same).sum())
    updated = int((~is_new & ~same).sum())
    return int(is_new.sum()), updated, unchanged


class ParquetStore:
    """One Parquet file per (vendor, freq, instrument_id), with atomic dedupe-append."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    # paths -----------------------------------------------------------------
    def series_dir(self, vendor: str, freq: str = "1d") -> Path:
        return self.root / f"vendor={vendor}" / f"freq={freq}"

    def path_for(self, instrument_id: str, vendor: str, freq: str = "1d") -> Path:
        if "/" in instrument_id or "\\" in instrument_id:
            raise StoreError(f"instrument_id {instrument_id!r} is not filesystem-safe")
        return self.series_dir(vendor, freq) / f"{instrument_id}.parquet"

    def exists(self, instrument_id: str, vendor: str, freq: str = "1d") -> bool:
        return self.path_for(instrument_id, vendor, freq).exists()

    def list_instruments(self, vendor: str, freq: str = "1d") -> list[str]:
        d = self.series_dir(vendor, freq)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.parquet"))

    # read ------------------------------------------------------------------
    def read_series(self, instrument_id: str, vendor: str, freq: str = "1d") -> pd.DataFrame:
        path = self.path_for(instrument_id, vendor, freq)
        if not path.exists():
            return _empty_bars()
        try:
            return _from_arrow(pq.read_table(path))
        except Exception as exc:  # noqa: BLE001 - corrupt file, schema drift, ...
            raise StoreError(f"cannot read {path}: {exc}") from exc

    def read_bars(
        self,
        instrument_ids: Iterable[str],
        vendor: str,
        start: date | str | None = None,
        end: date | str | None = None,
        freq: str = "1d",
    ) -> pd.DataFrame:
        frames = [self.read_series(iid, vendor, freq) for iid in instrument_ids]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return _empty_bars()
        out = pd.concat(frames, ignore_index=True)
        if start is not None:
            out = out[out["session_date"] >= pd.Timestamp(to_date(start))]
        if end is not None:
            out = out[out["session_date"] <= pd.Timestamp(to_date(end))]
        return out.sort_values(list(BAR_KEY), kind="mergesort").reset_index(drop=True)

    def last_session(self, instrument_id: str, vendor: str, freq: str = "1d") -> date | None:
        s = self.read_series(instrument_id, vendor, freq)
        return None if s.empty else s["session_date"].max().date()

    # write -----------------------------------------------------------------
    def write_bars(self, df: pd.DataFrame, freq: str = "1d") -> list[WriteResult]:
        """Dedupe-append each instrument's rows into its own file. `vendor` is taken from the frame."""
        bars = coerce_bars(df)
        if bars.empty:
            return []
        results: list[WriteResult] = []
        for (vendor, instrument_id), chunk in bars.groupby(["vendor", "instrument_id"], sort=True):
            results.append(self._write_series(str(instrument_id), str(vendor), freq, chunk))
        return results

    def _write_series(
        self, instrument_id: str, vendor: str, freq: str, incoming: pd.DataFrame
    ) -> WriteResult:
        path = self.path_for(instrument_id, vendor, freq)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(path) + ".lock")
        with lock:
            existing = self.read_series(instrument_id, vendor, freq)
            rows_new, rows_updated, rows_unchanged = _diff_counts(existing, incoming)
            merged = dedupe_bars(existing, incoming)
            tmp = path.with_name(path.name + ".tmp")
            try:
                pq.write_table(_to_arrow(merged), tmp, compression="zstd")
                os.replace(tmp, path)
            finally:
                if tmp.exists():
                    tmp.unlink()
        return WriteResult(
            instrument_id=instrument_id,
            path=path,
            rows_before=len(existing),
            rows_after=len(merged),
            rows_new=rows_new,
            rows_updated=rows_updated,
            rows_unchanged=rows_unchanged,
        )

    def delete_series(self, instrument_id: str, vendor: str, freq: str = "1d") -> bool:
        path = self.path_for(instrument_id, vendor, freq)
        if path.exists():
            path.unlink()
            return True
        return False


# ----------------------------------------------------------------------------- DuckDB store

SCHEMA_VERSION = 1

_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL,
        applied_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id VARCHAR PRIMARY KEY,
        symbol VARCHAR NOT NULL,
        asset_class VARCHAR NOT NULL,
        market VARCHAR NOT NULL,
        currency VARCHAR NOT NULL,
        name VARCHAR,
        price_scale DOUBLE NOT NULL,
        stamp_duty_class VARCHAR NOT NULL,
        tradable BOOLEAN NOT NULL,
        sector VARCHAR,
        country VARCHAR,
        lot_size INTEGER NOT NULL,
        identifiers VARCHAR NOT NULL,
        notes VARCHAR,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbol_history (
        instrument_id VARCHAR NOT NULL,
        vendor VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        valid_from DATE NOT NULL,
        valid_to DATE,
        recorded_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS series_meta (
        instrument_id VARCHAR NOT NULL,
        vendor VARCHAR NOT NULL,
        freq VARCHAR NOT NULL,
        first_session DATE,
        last_session DATE,
        rows INTEGER NOT NULL,
        last_checked_at TIMESTAMPTZ NOT NULL,
        last_written_at TIMESTAMPTZ,
        PRIMARY KEY (instrument_id, vendor, freq)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS corporate_actions (
        instrument_id VARCHAR NOT NULL,
        ex_date DATE NOT NULL,
        action_type VARCHAR NOT NULL,
        value DOUBLE NOT NULL,
        currency VARCHAR,
        vendor VARCHAR NOT NULL,
        recorded_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (instrument_id, ex_date, action_type, vendor)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_series (
        series_id VARCHAR NOT NULL,
        vendor VARCHAR NOT NULL,
        date DATE NOT NULL,
        value DOUBLE,
        ingested_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (series_id, vendor, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage (
        lineage_id VARCHAR PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL,
        vendor VARCHAR NOT NULL,
        operation VARCHAR NOT NULL,
        instrument_ids VARCHAR,
        params VARCHAR,
        rows INTEGER,
        sha256 VARCHAR,
        versions VARCHAR,
        diff VARCHAR,
        note VARCHAR
    )
    """,
)


class DuckDBStore:
    """Relational store. Readers use read-only connections; writers are serialised by a file lock."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = FileLock(str(self.path) + ".lock")

    # connections -----------------------------------------------------------
    @contextmanager
    def write(self) -> Iterator[duckdb.DuckDBPyConnection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            con = duckdb.connect(str(self.path))
            try:
                yield con
            finally:
                con.close()

    @contextmanager
    def read(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self.path.exists():
            raise StoreError(f"DuckDB file {self.path} does not exist; call migrate() first")
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            yield con
        finally:
            con.close()

    def migrate(self) -> int:
        """Create tables that are missing and record the schema version. Idempotent."""
        with self.write() as con:
            for ddl in _DDL:
                con.execute(ddl)
            current = con.execute("SELECT max(version) FROM schema_version").fetchone()[0]
            if current is None or current < SCHEMA_VERSION:
                con.execute("INSERT INTO schema_version VALUES (?, ?)", [SCHEMA_VERSION, now()])
        return SCHEMA_VERSION

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        with self.read() as con:
            return con.execute(sql, list(params or [])).df()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        with self.write() as con:
            con.execute(sql, list(params or []))

    def tables(self) -> list[str]:
        return sorted(self.query("SELECT table_name FROM information_schema.tables")["table_name"].tolist())

    # instruments -----------------------------------------------------------
    def upsert_instruments(self, instruments: Iterable[Instrument]) -> int:
        rows = [
            (
                i.instrument_id,
                i.symbol,
                i.asset_class.value,
                i.market.value,
                i.currency.value,
                i.name,
                float(i.price_scale),
                i.stamp_duty_class.value,
                bool(i.tradable),
                i.sector,
                i.country,
                int(i.lot_size),
                json.dumps(i.identifiers, sort_keys=True),
                i.notes,
                now(),
            )
            for i in instruments
        ]
        if not rows:
            return 0
        with self.write() as con:
            con.executemany(
                "INSERT OR REPLACE INTO instruments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    @staticmethod
    def _row_to_instrument(row: dict[str, Any]) -> Instrument:
        def text(key: str) -> str | None:
            value = row.get(key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return str(value)

        return Instrument(
            instrument_id=row["instrument_id"],
            symbol=row["symbol"],
            asset_class=row["asset_class"],
            market=row["market"],
            currency=row["currency"],
            name=text("name"),
            price_scale=float(row["price_scale"]),
            stamp_duty_class=row["stamp_duty_class"],
            tradable=bool(row["tradable"]),
            sector=text("sector"),
            country=text("country"),
            lot_size=int(row["lot_size"]),
            identifiers=json.loads(row["identifiers"] or "{}"),
            notes=text("notes"),
        )

    def get_instrument(self, instrument_id: str) -> Instrument | None:
        df = self.query("SELECT * FROM instruments WHERE instrument_id = ?", [instrument_id])
        if df.empty:
            return None
        return self._row_to_instrument(df.iloc[0].to_dict())

    def list_instruments(self) -> list[Instrument]:
        df = self.query("SELECT * FROM instruments ORDER BY instrument_id")
        return [self._row_to_instrument(r) for r in df.to_dict("records")]

    # series metadata -------------------------------------------------------
    def upsert_series_meta(
        self,
        instrument_id: str,
        vendor: str,
        freq: str,
        *,
        first_session: date | None,
        last_session: date | None,
        rows: int,
        written: bool,
    ) -> None:
        ts = now()
        with self.write() as con:
            prev = con.execute(
                "SELECT last_written_at FROM series_meta WHERE instrument_id = ? AND vendor = ? AND freq = ?",
                [instrument_id, vendor, freq],
            ).fetchone()
            last_written = ts if written else (prev[0] if prev else None)
            con.execute(
                "INSERT OR REPLACE INTO series_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [instrument_id, vendor, freq, first_session, last_session, int(rows), ts, last_written],
            )

    def get_series_meta(self, instrument_id: str, vendor: str, freq: str) -> dict[str, Any] | None:
        df = self.query(
            "SELECT * FROM series_meta WHERE instrument_id = ? AND vendor = ? AND freq = ?",
            [instrument_id, vendor, freq],
        )
        if df.empty:
            return None
        meta = df.iloc[0].to_dict()
        for key in ("first_session", "last_session"):  # DuckDB DATE -> pandas Timestamp -> date
            value = meta.get(key)
            meta[key] = None if value is None or pd.isna(value) else pd.Timestamp(value).date()
        return meta

    # corporate actions -----------------------------------------------------
    def upsert_corporate_actions(self, df: pd.DataFrame | None) -> int:
        if df is None or df.empty:
            return 0
        rows = []
        for r in df.itertuples(index=False):
            currency = r.currency if isinstance(r.currency, str) else None
            rows.append(
                (
                    str(r.instrument_id),
                    pd.Timestamp(r.ex_date).date(),
                    str(r.action_type),
                    float(r.value),
                    currency,
                    str(r.vendor),
                    pd.Timestamp(r.recorded_at).to_pydatetime(),
                )
            )
        with self.write() as con:
            con.executemany("INSERT OR REPLACE INTO corporate_actions VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def load_corporate_actions(
        self,
        instrument_ids: Sequence[str],
        vendor: str | None = None,
        known_by: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Actions for the instruments; `known_by` keeps only rows recorded at or before that time."""
        ids = list(instrument_ids)
        if not ids:
            return _empty_actions()
        sql = f"SELECT * FROM corporate_actions WHERE instrument_id IN ({', '.join('?' * len(ids))})"
        params: list[Any] = list(ids)
        if vendor:
            sql += " AND vendor = ?"
            params.append(vendor)
        if known_by is not None:
            sql += " AND recorded_at <= ?"
            params.append(pd.Timestamp(known_by).to_pydatetime())
        df = self.query(sql + " ORDER BY instrument_id, ex_date, action_type", params)
        if df.empty:
            return _empty_actions()
        df["ex_date"] = pd.to_datetime(df["ex_date"]).astype("datetime64[ns]")
        df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True).astype("datetime64[ns, UTC]")
        df["value"] = df["value"].astype("float64")
        df["currency"] = pd.Series([c if isinstance(c, str) else None for c in df["currency"]], dtype=object)
        return df.reset_index(drop=True)

    # macro series ----------------------------------------------------------
    def upsert_macro(self, df: pd.DataFrame | None) -> int:
        if df is None or df.empty:
            return 0
        rows = [
            (
                str(r.series_id),
                str(r.vendor),
                pd.Timestamp(r.date).date(),
                None if pd.isna(r.value) else float(r.value),
                pd.Timestamp(r.ingested_at).to_pydatetime(),
            )
            for r in df.itertuples(index=False)
        ]
        with self.write() as con:
            con.executemany("INSERT OR REPLACE INTO macro_series VALUES (?, ?, ?, ?, ?)", rows)
        return len(rows)

    def load_macro(
        self, series_id: str, vendor: str | None = None, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        sql = "SELECT * FROM macro_series WHERE series_id = ?"
        params: list[Any] = [series_id]
        if vendor:
            sql += " AND vendor = ?"
            params.append(vendor)
        if start is not None:
            sql += " AND date >= ?"
            params.append(start)
        if end is not None:
            sql += " AND date <= ?"
            params.append(end)
        df = self.query(sql + " ORDER BY date", params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
            df["value"] = df["value"].astype("float64")
        return df

    def macro_last_date(self, series_id: str, vendor: str) -> date | None:
        df = self.query(
            "SELECT max(date) AS d FROM macro_series WHERE series_id = ? AND vendor = ?", [series_id, vendor]
        )
        value = df["d"].iloc[0]
        return None if value is None or pd.isna(value) else pd.Timestamp(value).date()

    # parquet view ----------------------------------------------------------
    def register_bars_view(self, parquet_root: Path | str, name: str = "bars", freq: str = "1d") -> bool:
        """Expose every Parquet series of one frequency as a DuckDB view. False when no files exist yet.

        The files already carry a `vendor` column, so hive partitioning is off (it would clash).
        """
        root = Path(parquet_root)
        pattern = root / "vendor=*" / f"freq={freq}" / "*.parquet"
        if not list(root.glob(f"vendor=*/freq={freq}/*.parquet")):
            return False
        glob = str(pattern).replace("'", "''")
        with self.write() as con:
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{glob}', hive_partitioning = false, union_by_name = true)"
            )
        return True
