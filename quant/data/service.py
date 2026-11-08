"""DataService: the single entry point for prices. adapter -> store -> lineage -> serve.

`get_bars()` serves from the local Parquet cache and fetches only what is missing: an older head if `start`
precedes the cached range, or a newer tail when the cache ends before the latest *complete* session and the
series has not been checked within `data.cache_ttl_hours`. Every fetch leaves a lineage row.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from quant.config.settings import QuantSettings
from quant.core.calendars import latest_complete_session
from quant.core.catalog import InstrumentCatalog, load_default_catalog
from quant.core.errors import AdapterError, ConfigError
from quant.core.time import now, to_date, today
from quant.core.types import Instrument
from quant.data.adapters.base import Capability, MarketDataAdapter
from quant.data.lineage import LineageLog, LineageRecord, frame_digest
from quant.data.store import DuckDBStore, ParquetStore, WriteResult


@dataclass
class FetchReport:
    vendor: str
    start: date
    end: date
    requested: list[str]
    rows: int
    writes: list[WriteResult] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    corporate_actions: int = 0
    lineage_id: str | None = None

    @property
    def rows_new(self) -> int:
        return sum(w.rows_new for w in self.writes)

    @property
    def rows_updated(self) -> int:
        return sum(w.rows_updated for w in self.writes)

    def summary(self) -> str:
        return (
            f"{self.vendor} {self.start}..{self.end}: {self.rows} rows for "
            f"{len(self.writes)}/{len(self.requested)} instrument(s) "
            f"(+{self.rows_new} new, {self.rows_updated} restated, {self.corporate_actions} actions); "
            f"{len(self.errors)} error(s)"
        )


class DataService:
    def __init__(
        self,
        settings: QuantSettings,
        catalog: InstrumentCatalog | None = None,
        adapters: dict[str, MarketDataAdapter] | None = None,
        freq: str = "1d",
    ) -> None:
        self.settings = settings
        settings.paths.ensure()
        self.freq = freq
        self.parquet = ParquetStore(settings.paths.parquet_dir)
        self.db = DuckDBStore(settings.paths.duckdb_path)
        self.db.migrate()
        self.lineage = LineageLog(self.db)
        self.catalog = catalog if catalog is not None else load_default_catalog()
        self.db.upsert_instruments(self.catalog)
        self.adapters: dict[str, MarketDataAdapter] = (
            adapters if adapters is not None else self._default_adapters()
        )

    # ------------------------------------------------------------------ adapters
    def _default_adapters(self) -> dict[str, MarketDataAdapter]:
        from quant.data.adapters.yfinance_adapter import YFinanceAdapter

        return {
            "yfinance": YFinanceAdapter(drop_unfinished_session=self.settings.data.drop_unfinished_session),
        }

    def adapter_for(self, capability: Capability | str, vendor: str | None = None) -> MarketDataAdapter:
        cap = Capability(capability)
        names = [vendor] if vendor else list(self.settings.data.adapter_priority)
        for name in names:
            adapter = self.adapters.get(name)
            if adapter is not None and cap in adapter.capabilities():
                return adapter
        raise AdapterError(f"no configured adapter provides {cap.value}; tried {names}")

    # ------------------------------------------------------------------ instruments
    def resolve(self, ref: str | Instrument) -> Instrument:
        """Instrument by id, canonical symbol (if unique) or any vendor symbol."""
        if isinstance(ref, Instrument):
            return ref
        inst = self.catalog.get(ref)
        if inst is not None:
            return inst
        by_symbol = self.catalog.by_symbol(ref)
        if len(by_symbol) == 1:
            return by_symbol[0]
        if len(by_symbol) > 1:
            ids = ", ".join(i.instrument_id for i in by_symbol)
            raise ConfigError(f"symbol {ref!r} is ambiguous; use an instrument_id: {ids}")
        for vendor in ("yahoo", "stooq", "fred", "boe", "bbg", "ric", "isin", "sedol", "figi", "alpaca"):
            inst = self.catalog.by_vendor_symbol(vendor, ref)
            if inst is not None:
                return inst
        raise ConfigError(f"unknown instrument {ref!r}; add it to the catalogue first")

    def register(self, instruments: Iterable[Instrument]) -> int:
        count = 0
        for inst in instruments:
            self.catalog.add(inst, replace=True)
            count += 1
        self.db.upsert_instruments(instruments)
        return count

    # ------------------------------------------------------------------ fetch
    def fetch_bars(
        self, instruments: Sequence[Instrument | str], start: date, end: date, vendor: str | None = None
    ) -> FetchReport:
        insts = [self.resolve(i) for i in instruments]
        adapter = self.adapter_for(Capability.BARS_DAILY, vendor)
        start_d, end_d = to_date(start), to_date(end)
        result = adapter.get_bars(insts, start_d, end_d, self.freq)
        writes = self.parquet.write_bars(result.frame, self.freq) if not result.frame.empty else []
        actions = result.extras.get("corporate_actions")
        n_actions = 0 if actions is None else len(actions)
        record = LineageRecord(
            vendor=adapter.name,
            operation="fetch",
            instrument_ids=result.requested,
            params={**result.params, "freq": self.freq},
            rows=len(result.frame),
            sha256=frame_digest(result.frame) if not result.frame.empty else None,
            diff={"writes": [w.as_dict() for w in writes], "errors": result.errors, "actions": n_actions},
        )
        self.lineage.record(record)
        written_ids = {w.instrument_id for w in writes}
        for inst in insts:
            series = self.parquet.read_series(inst.instrument_id, adapter.name, self.freq)
            self.db.upsert_series_meta(
                inst.instrument_id,
                adapter.name,
                self.freq,
                first_session=None if series.empty else series["session_date"].min().date(),
                last_session=None if series.empty else series["session_date"].max().date(),
                rows=len(series),
                written=inst.instrument_id in written_ids,
            )
        return FetchReport(
            vendor=adapter.name,
            start=start_d,
            end=end_d,
            requested=result.requested,
            rows=len(result.frame),
            writes=writes,
            errors=result.errors,
            corporate_actions=n_actions,
            lineage_id=record.lineage_id,
        )

    # ------------------------------------------------------------------ serve
    def get_bars(
        self,
        instruments: Sequence[Instrument | str],
        start: date | str,
        end: date | str | None = None,
        vendor: str | None = None,
        refresh: bool | None = None,
    ) -> pd.DataFrame:
        """Serve bars from the cache, fetching only a missing head/tail; refresh=False never fetches."""
        insts = [self.resolve(i) for i in instruments]
        start_d = to_date(start)
        end_d = to_date(end) if end is not None else today()
        adapter = self.adapter_for(Capability.BARS_DAILY, vendor)
        if refresh is not False:
            plan: dict[tuple[date, date], list[Instrument]] = {}
            for inst in insts:
                for rng in self._missing_ranges(inst, start_d, end_d, adapter.name, force=bool(refresh)):
                    plan.setdefault(rng, []).append(inst)
            for (s, e), group in sorted(plan.items()):
                self.fetch_bars(group, s, e, vendor=adapter.name)
        ids = [i.instrument_id for i in insts]
        return self.parquet.read_bars(ids, adapter.name, start_d, end_d, self.freq)

    def _missing_ranges(
        self, inst: Instrument, start: date, end: date, vendor: str, *, force: bool
    ) -> list[tuple[date, date]]:
        series = self.parquet.read_series(inst.instrument_id, vendor, self.freq)
        if series.empty:
            return [(start, end)]
        first = series["session_date"].min().date()
        last = series["session_date"].max().date()
        ranges: list[tuple[date, date]] = []
        if start < first:
            ranges.append((start, first - timedelta(days=1)))
        expected_last = latest_complete_session(inst.calendar_code)
        if end < expected_last:
            expected_last = end
        if last < expected_last and (force or self._is_stale(inst.instrument_id, vendor)):
            ranges.append((last + timedelta(days=1), end))
        return ranges

    def _is_stale(self, instrument_id: str, vendor: str) -> bool:
        meta = self.db.get_series_meta(instrument_id, vendor, self.freq)
        if meta is None or meta.get("last_checked_at") is None:
            return True
        checked = pd.Timestamp(meta["last_checked_at"])
        checked = checked.tz_localize("UTC") if checked.tzinfo is None else checked.tz_convert("UTC")
        ttl = pd.Timedelta(hours=float(self.settings.data.cache_ttl_hours))
        return pd.Timestamp(now()) - checked >= ttl

    # ------------------------------------------------------------------ introspection
    def cached_series(self, vendor: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM series_meta"
        params: list[Any] = []
        if vendor:
            sql += " WHERE vendor = ?"
            params.append(vendor)
        return self.db.query(sql + " ORDER BY instrument_id", params)
