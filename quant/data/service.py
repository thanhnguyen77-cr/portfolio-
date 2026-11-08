"""DataService: the single entry point for prices. adapter -> store -> lineage -> serve.

`get_bars()` serves from the local Parquet cache and fetches only what is missing: an older head if `start`
precedes the cached range, or a newer tail when the cache ends before the latest *complete* session and the
series has not been checked within `data.cache_ttl_hours`. Every fetch leaves a lineage row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
from quant.data.corporate_actions import adjust_bars
from quant.data.lineage import LineageLog, LineageRecord, frame_digest
from quant.data.store import DuckDBStore, ParquetStore, WriteResult
from quant.data.validation import DataQualityReport, ValidationConfig, validate_bars
from quant.universe.pit import PITUniverse


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
        from quant.data.adapters.boe import BankOfEnglandAdapter
        from quant.data.adapters.fred import FREDAdapter
        from quant.data.adapters.yfinance_adapter import YFinanceAdapter

        fred_key = self.settings.fred_api_key.get_secret_value() if self.settings.fred_api_key else None
        return {
            "yfinance": YFinanceAdapter(drop_unfinished_session=self.settings.data.drop_unfinished_session),
            "fred": FREDAdapter(api_key=fred_key),
            "boe": BankOfEnglandAdapter(),
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
        n_actions = self.db.upsert_corporate_actions(actions)
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

    def _is_stale(self, instrument_id: str, vendor: str, freq: str | None = None) -> bool:
        meta = self.db.get_series_meta(instrument_id, vendor, freq or self.freq)
        if meta is None or meta.get("last_checked_at") is None:
            return True
        checked = pd.Timestamp(meta["last_checked_at"])
        checked = checked.tz_localize("UTC") if checked.tzinfo is None else checked.tz_convert("UTC")
        ttl = pd.Timedelta(hours=float(self.settings.data.cache_ttl_hours))
        return pd.Timestamp(now()) - checked >= ttl

    # ------------------------------------------------------------------ adjusted bars & validation
    def get_corporate_actions(
        self, instruments: Sequence[Instrument | str], vendor: str | None = None, known_by=None
    ) -> pd.DataFrame:  # noqa: ANN001 - known_by: timestamp-like
        ids = [self.resolve(i).instrument_id for i in instruments]
        return self.db.load_corporate_actions(ids, vendor=vendor, known_by=known_by)

    def get_adjusted_bars(
        self,
        instruments: Sequence[Instrument | str],
        start: date | str,
        end: date | str | None = None,
        *,
        as_of: date | str | None = None,
        vendor: str | None = None,
        refresh: bool | None = None,
    ) -> pd.DataFrame:
        """Bars plus adj_* columns, adjusted only for actions with ex_date <= as_of (default: all)."""
        bars = self.get_bars(instruments, start, end, vendor=vendor, refresh=refresh)
        if bars.empty:
            return adjust_bars(bars, bars)
        ids = sorted(bars["instrument_id"].unique())
        actions = self.db.load_corporate_actions(ids, vendor=bars["vendor"].iloc[0])
        return adjust_bars(bars, actions, as_of=as_of)

    def validate(
        self,
        instruments: Sequence[Instrument | str],
        start: date | str,
        end: date | str | None = None,
        *,
        vendor: str | None = None,
        refresh: bool | None = False,
        config: ValidationConfig | None = None,
    ) -> DataQualityReport:
        """Data-quality report for cached bars (refresh=False by default: validation never fetches)."""
        insts = [self.resolve(i) for i in instruments]
        bars = self.get_bars(insts, start, end, vendor=vendor, refresh=refresh)
        by_id = {i.instrument_id: i for i in insts}
        actions = self.db.load_corporate_actions(list(by_id)) if not bars.empty else None
        report = validate_bars(bars, by_id, actions, config)
        if not bars.empty:
            self.lineage.record(
                LineageRecord(
                    vendor=bars["vendor"].iloc[0],
                    operation="validate",
                    instrument_ids=list(by_id),
                    rows=len(bars),
                    diff={iid: {"score": q.score, "grade": q.grade} for iid, q in report.results.items()},
                )
            )
        return report

    # ------------------------------------------------------------------ macro series
    def macro_vendor_for(self, series_id: str, vendor: str | None = None) -> str:
        if vendor:
            return vendor
        for name in ("fred", "boe"):
            inst = self.catalog.by_vendor_symbol(name, series_id)
            if inst is not None and name in self.adapters:
                return name
        return self.adapter_for(Capability.MACRO).name

    def fetch_macro(
        self, series_ids: Sequence[str], start: date, end: date, vendor: str | None = None
    ) -> FetchReport:
        vendor_name = self.macro_vendor_for(series_ids[0], vendor)
        adapter = self.adapter_for(Capability.MACRO, vendor_name)
        start_d, end_d = to_date(start), to_date(end)
        result = adapter.get_macro_series(list(series_ids), start_d, end_d)
        rows = self.db.upsert_macro(result.frame)
        record = LineageRecord(
            vendor=adapter.name,
            operation="fetch_macro",
            instrument_ids=list(series_ids),
            params={**result.params},
            rows=rows,
            sha256=frame_digest(result.frame) if not result.frame.empty else None,
            diff={"errors": result.errors},
        )
        self.lineage.record(record)
        for sid in series_ids:
            cached = self.db.load_macro(sid, adapter.name)
            self.db.upsert_series_meta(
                sid,
                adapter.name,
                "macro",
                first_session=None if cached.empty else cached["date"].min().date(),
                last_session=None if cached.empty else cached["date"].max().date(),
                rows=len(cached),
                written=sid in set(result.frame["series_id"]) if not result.frame.empty else False,
            )
        return FetchReport(
            vendor=adapter.name,
            start=start_d,
            end=end_d,
            requested=list(series_ids),
            rows=rows,
            errors=result.errors,
            lineage_id=record.lineage_id,
        )

    def get_macro(
        self,
        series_id: str,
        start: date | str,
        end: date | str | None = None,
        vendor: str | None = None,
        refresh: bool | None = None,
    ) -> pd.Series:
        """A macro series as a date-indexed Series, fetching a missing head/tail like get_bars."""
        vendor_name = self.macro_vendor_for(series_id, vendor)
        start_d = to_date(start)
        end_d = to_date(end) if end is not None else today()
        if refresh is not False:
            cached = self.db.load_macro(series_id, vendor_name)
            if cached.empty:
                self.fetch_macro([series_id], start_d, end_d, vendor_name)
            else:
                first, last = cached["date"].min().date(), cached["date"].max().date()
                if start_d < first:
                    self.fetch_macro([series_id], start_d, first - timedelta(days=1), vendor_name)
                stale = refresh or self._is_stale(series_id, vendor_name, freq="macro")
                if last < end_d and stale:
                    self.fetch_macro([series_id], last + timedelta(days=1), end_d, vendor_name)
        df = self.db.load_macro(series_id, vendor_name, start_d, end_d)
        return pd.Series(df["value"].to_numpy(), index=pd.DatetimeIndex(df["date"]), name=series_id)

    # ------------------------------------------------------------------ universes
    def universe(self, name: str) -> PITUniverse:
        return PITUniverse(self.db, name)

    def build_universe(
        self,
        name: str,
        *,
        html: str | None = None,
        changes_html: str | None = None,
        as_of: date | None = None,
        reconstruct: bool = True,
        expected_rows: tuple[int, int] | None = None,
        source=None,
        renames: Mapping[str, str] | None = None,
    ) -> tuple[PITUniverse, dict[str, Any]]:  # noqa: ANN001 - source: WikipediaSource
        """Fetch constituents, register instruments, record a snapshot and (S&P) reconstruct history."""
        from quant.universe.constituents import UNIVERSES, load_snapshot, members_to_instruments

        snap = load_snapshot(
            name, source, html=html, changes_html=changes_html, as_of=as_of, expected_rows=expected_rows
        )
        spec = UNIVERSES[name]
        instruments = members_to_instruments(snap.members, spec, self.catalog)
        self.register(instruments)
        ids = [i.instrument_id for i in instruments]
        uni = self.universe(name)
        delta = uni.add_snapshot(snap.as_of, ids, source=f"wikipedia:{snap.html_sha256[:12]}")
        info: dict[str, Any] = {
            "members": len(ids),
            "added": list(delta.added),
            "removed": list(delta.removed),
            "html_sha256": snap.html_sha256,
            "notes": list(snap.notes),
            "reconstructed_intervals": 0,
        }
        if reconstruct and snap.changes is not None and not snap.changes.empty:
            by_symbol = {i.symbol: i.instrument_id for i in instruments}
            ch = snap.changes.copy()
            aliases = dict(renames or {})
            ch["added_id"] = ch["added"].map(
                lambda s: self._resolve_or_create_us(s, spec.market, by_symbol, aliases)
            )
            ch["removed_id"] = ch["removed"].map(
                lambda s: self._resolve_or_create_us(s, spec.market, by_symbol, aliases)
            )
            info["reconstructed_intervals"] = uni.reconstruct_from_changes(
                ids, snap.as_of, ch, source="wikipedia_changes"
            )
        self.lineage.record(
            LineageRecord(
                vendor="wikipedia",
                operation="universe",
                instrument_ids=ids,
                rows=len(ids),
                sha256=snap.html_sha256,
                params={"universe": name, "as_of": str(snap.as_of)},
                diff={k: v for k, v in info.items() if k != "notes"},
            )
        )
        info["quality"] = uni.quality()
        return uni, info

    def _resolve_or_create_us(self, symbol, market, by_symbol, aliases=None):  # noqa: ANN001
        if not isinstance(symbol, str) or not symbol:
            return None
        original = symbol
        symbol = (aliases or {}).get(symbol, symbol)
        if symbol in by_symbol:
            if original != symbol and self.db.lookup_alias("yahoo", original) is None:
                self.db.add_symbol_alias(by_symbol[symbol], "yahoo", original)
            return by_symbol[symbol]
        alias_hit = self.db.lookup_alias("yahoo", symbol)
        if alias_hit is not None:
            by_symbol[symbol] = alias_hit
            return alias_hit
        existing = self.catalog.by_vendor_symbol("yahoo", symbol) or next(
            iter(self.catalog.by_symbol(symbol)), None
        )
        if existing is not None:
            by_symbol[symbol] = existing.instrument_id
            return existing.instrument_id
        inst = Instrument(
            symbol=symbol,
            asset_class="equity",
            market=market,
            currency="USD",
            country="US",
            stamp_duty_class="exempt",
            identifiers={"yahoo": symbol},
            notes="former index member from the changes table; may be delisted or renamed",
        )
        self.register([inst])
        by_symbol[symbol] = inst.instrument_id
        return inst.instrument_id

    def screen(
        self,
        instruments: Sequence[Instrument | str],
        start: date | str,
        end: date | str | None = None,
        *,
        config=None,
        refresh: bool | None = None,
        validate: bool = True,
    ) -> pd.DataFrame:  # noqa: ANN001 - config: ScreenConfig
        """Liquidity / volatility / data-quality screen over cached bars (fetching per `refresh`)."""
        from quant.data.fx import FXRates
        from quant.universe.screen import screen_universe

        insts = [self.resolve(i) for i in instruments]
        bars = self.get_bars(insts, start, end, refresh=refresh)
        base = self.settings.trading.base_currency.value
        fx = None
        foreign = sorted({i.currency.value for i in insts} - {base})
        if foreign:
            needed = {frozenset({base, f}) for f in foreign}
            pairs = [
                p
                for p in self.catalog.filter(asset_class="fx")
                if frozenset({p.symbol[:3], p.symbol[3:]}) in needed
            ]
            fx_bars = self.get_bars(pairs, start, end, refresh=refresh) if pairs else bars.iloc[0:0]
            fx = FXRates.from_bars(fx_bars, self.catalog)
        dq = validate_bars(bars, {i.instrument_id: i for i in insts}) if validate and not bars.empty else None
        return screen_universe(
            bars,
            {i.instrument_id: i for i in insts},
            base_currency=base,
            fx=fx,
            data_quality=dq,
            config=config,
        )

    # ------------------------------------------------------------------ introspection
    def cached_series(self, vendor: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM series_meta"
        params: list[Any] = []
        if vendor:
            sql += " WHERE vendor = ?"
            params.append(vendor)
        return self.db.query(sql + " ORDER BY instrument_id", params)
