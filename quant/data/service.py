"""DataService: the single entry point for prices. adapter -> store -> lineage -> serve.

`get_bars()` serves from the local Parquet cache and fetches only what is missing: an older head if `start`
precedes the cached range, or a newer tail when the cache ends before the latest *complete* session and the
series has not been checked within `data.cache_ttl_hours`. Every fetch leaves a lineage row.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
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
        from quant.data.adapters.ons import ONSAdapter
        from quant.data.adapters.sec_edgar import SECEdgarAdapter
        from quant.data.adapters.yfinance_adapter import YFinanceAdapter

        fred_key = self.settings.fred_api_key.get_secret_value() if self.settings.fred_api_key else None
        return {
            "yfinance": YFinanceAdapter(drop_unfinished_session=self.settings.data.drop_unfinished_session),
            "fred": FREDAdapter(api_key=fred_key),
            "boe": BankOfEnglandAdapter(),
            "ons": ONSAdapter(),
            "sec_edgar": SECEdgarAdapter(user_agent=self.settings.sec_user_agent),
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
        if "/" in series_id and "ons" in self.adapters:  # ONS ids are "<dataset>/<series>"
            return "ons"
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

    def macro_series(
        self,
        name: str,
        start: date | str,
        end: date | str | None = None,
        refresh: bool | None = None,
    ) -> pd.Series:
        """A catalogued macro series (see quant.macro.catalog) by friendly name, e.g. 'us_cpi' or 'uk_10y'."""
        from quant.macro.catalog import macro_series_spec

        try:
            spec = macro_series_spec(name)
        except KeyError as exc:
            raise ConfigError(str(exc)) from None
        return self.get_macro(spec.series_id, start, end, vendor=spec.vendor, refresh=refresh).rename(name)

    def macro_dashboard(
        self,
        region: str = "us",
        start: date | str | None = None,
        end: date | str | None = None,
        *,
        refresh: bool | None = False,
    ) -> dict[str, Any]:
        """Cross-asset regime board for 'us' or 'uk' from whatever inputs are available (missing ones listed).

        Inputs that cannot be served (not cached and refresh=False, vendor error, empty response) are reported
        in `notes` and the indicator is listed under `missing`; nothing is imputed.
        """
        from quant.macro import annualised_3m, cross_asset_dashboard, slope, yoy

        end_d = to_date(end) if end is not None else today()
        start_d = to_date(start) if start is not None else end_d - timedelta(days=365 * 6)
        series: dict[str, pd.Series] = {}
        notes: list[str] = []

        def grab(name: str) -> pd.Series | None:
            try:
                s = self.macro_series(name, start_d, end_d, refresh=refresh)
            except Exception as exc:  # noqa: BLE001 - one failed input must not sink the board
                notes.append(f"{name}: {type(exc).__name__}: {str(exc)[:120]}")
                return None
            if s.dropna().empty:
                notes.append(f"{name}: no observations available ({start_d}..{end_d}, refresh={refresh})")
                return None
            return s

        def closes(ref: str) -> pd.Series | None:
            try:
                bars = self.get_bars([ref], start_d, end_d, refresh=refresh)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{ref}: {type(exc).__name__}: {str(exc)[:120]}")
                return None
            if bars.empty:
                notes.append(f"{ref}: no cached bars ({start_d}..{end_d}, refresh={refresh})")
                return None
            return bars.set_index("session_date")["close"].astype(float)

        def put(key: str, s: pd.Series | None) -> None:
            if s is not None and not s.dropna().empty:
                series[key] = s

        if region == "us":
            ip = grab("us_industrial_production")
            put("growth_ip_yoy", yoy(ip) if ip is not None else None)
            put("growth_unemployment", grab("us_unemployment"))
            payrolls = grab("us_payrolls")
            put("growth_payrolls_3m", payrolls.diff(3) if payrolls is not None else None)
            cpi = grab("us_cpi")
            put("inflation_yoy", yoy(cpi) if cpi is not None else None)
            put("inflation_3m_ann", annualised_3m(cpi) if cpi is not None else None)
            put("rates_policy", grab("us_fed_funds"))
            y10, y2 = grab("us_10y"), grab("us_2y")
            put("rates_10y", y10)
            put("rates_slope_10y_2y", slope(y10, y2) if y10 is not None and y2 is not None else None)
            put("rates_real_10y", grab("us_real_10y"))
            put("credit_hy_oas", grab("us_hy_oas"))
            put("credit_ig_oas", grab("us_ig_oas"))
            put("fx_dollar", grab("us_dollar_index"))
            spx = closes("XNYS_SPX")
            put("equity_vs_200d", spx / spx.rolling(200).mean() - 1 if spx is not None else None)
            put("vol_index", closes("XNYS_VIX"))
        elif region == "uk":
            gdp = grab("uk_gdp_monthly")
            put("growth_ip_yoy", yoy(gdp) if gdp is not None else None)
            put("growth_unemployment", grab("uk_unemployment"))
            put("inflation_yoy", grab("uk_cpih"))
            bank_rate = grab("uk_bank_rate")
            put("rates_policy", bank_rate)
            y10 = grab("uk_10y")
            put("rates_10y", y10)
            if y10 is not None and bank_rate is not None:
                put("rates_slope_10y_2y", slope(y10, bank_rate))
                notes.append(
                    "rates_slope_10y_2y for the UK is 10y gilt minus Bank Rate (no cached 2y gilt series)"
                )
            put("fx_gbpusd", closes("FX_GBPUSD"))
            ukx = closes("XLON_UKX")
            put("equity_vs_200d", ukx / ukx.rolling(200).mean() - 1 if ukx is not None else None)
        else:
            raise ConfigError("region must be 'us' or 'uk'")

        board = cross_asset_dashboard(series, as_of=pd.Timestamp(end_d))
        board["notes"] = notes
        board["region"] = region
        board["as_of"] = end_d
        return board

    # ------------------------------------------------------------------ fundamentals & earnings
    def fetch_fundamentals(
        self, instruments: Sequence[Instrument | str], vendor: str | None = None
    ) -> FetchReport:
        insts = [self.resolve(i) for i in instruments]
        adapter = self.adapter_for(Capability.FUNDAMENTALS, vendor)
        result = adapter.get_fundamentals(insts)
        rows = self.db.upsert_fundamentals(result.frame)
        record = LineageRecord(
            vendor=adapter.name,
            operation="fetch_fundamentals",
            instrument_ids=result.requested,
            params=dict(result.params),
            rows=rows,
            diff={"errors": result.errors},
        )
        self.lineage.record(record)
        return FetchReport(
            vendor=adapter.name,
            start=today(),
            end=today(),
            requested=result.requested,
            rows=rows,
            errors=result.errors,
            lineage_id=record.lineage_id,
        )

    def get_fundamentals(
        self,
        instrument: Instrument | str,
        *,
        as_of: date | str | None = None,
        vendor: str | None = None,
        refresh: bool | None = None,
    ):  # noqa: ANN201 - StatementSet
        """Point-in-time statements from the cache (fetching once when empty unless refresh=False)."""
        from quant.fundamental.statements import StatementSet

        inst = self.resolve(instrument)
        rows = self.db.load_fundamentals([inst.instrument_id], vendor=vendor)
        if (rows.empty and refresh is not False) or refresh:
            self.fetch_fundamentals([inst], vendor=vendor)
            rows = self.db.load_fundamentals([inst.instrument_id], vendor=vendor)
        return StatementSet.from_rows(rows, inst.instrument_id).as_of(as_of)

    def fetch_earnings(
        self, instruments: Sequence[Instrument | str], vendor: str | None = None
    ) -> FetchReport:
        insts = [self.resolve(i) for i in instruments]
        adapter = self.adapter_for(Capability.EARNINGS, vendor)
        result = adapter.get_earnings(insts)
        rows = self.db.upsert_earnings(result.frame)
        record = LineageRecord(
            vendor=adapter.name,
            operation="fetch_earnings",
            instrument_ids=result.requested,
            rows=rows,
            diff={"errors": result.errors},
        )
        self.lineage.record(record)
        return FetchReport(
            vendor=adapter.name,
            start=today(),
            end=today(),
            requested=result.requested,
            rows=rows,
            errors=result.errors,
            lineage_id=record.lineage_id,
        )

    def get_earnings(
        self, instrument: Instrument | str, *, as_of: date | None = None, refresh: bool | None = None
    ) -> pd.DataFrame:
        inst = self.resolve(instrument)
        rows = self.db.load_earnings(inst.instrument_id, as_of=as_of)
        if (rows.empty and refresh is not False) or refresh:
            self.fetch_earnings([inst])
            rows = self.db.load_earnings(inst.instrument_id, as_of=as_of)
        return rows

    def last_close(self, instrument: Instrument | str, as_of: date | str | None = None) -> float | None:
        """Latest cached close at or before `as_of` (no fetching)."""
        inst = self.resolve(instrument)
        adapter = self.adapter_for(Capability.BARS_DAILY)
        series = self.parquet.read_series(inst.instrument_id, adapter.name, self.freq)
        if series.empty:
            return None
        if as_of is not None:
            series = series[series["session_date"] <= pd.Timestamp(to_date(as_of))]
        return None if series.empty else float(series["close"].iloc[-1])

    # ------------------------------------------------------------------ factors
    def build_factor_returns(
        self,
        instruments: Sequence[Instrument | str],
        start: date | str,
        end: date | str | None = None,
        *,
        factors: Sequence[str] = ("momentum", "low_vol", "size", "value", "quality"),
        rebalance: str = "monthly",
        n_quantiles: int = 5,
        sector_neutral: bool = False,
        refresh: bool | None = None,
    ):  # noqa: ANN201 - FactorReturns
        """Long-short factor returns over cached data. Inputs that are missing are skipped and reported."""
        from quant.factors import (
            factor_returns,
            fundamental_characteristics,
            price_characteristics,
            rank_normalise,
        )
        from quant.factors.construction import factor_scores
        from quant.fundamental import StatementSet, compute_ratios

        insts = [self.resolve(i) for i in instruments]
        ids = [i.instrument_id for i in insts]
        start_d = to_date(start)
        end_d = to_date(end) if end is not None else today()
        bars = self.get_adjusted_bars(insts, start_d - timedelta(days=420), end_d, refresh=refresh)
        if bars.empty:
            raise ConfigError("no cached bars for the requested instruments")
        closes = bars.pivot(index="session_date", columns="instrument_id", values="adj_close").sort_index()
        notes: list[str] = []
        wanted = list(factors)
        stmts = {iid: StatementSet.from_rows(self.db.load_fundamentals([iid]), iid) for iid in ids}
        with_fund = [iid for iid, s in stmts.items() if not s.rows.empty]
        if any(f in wanted for f in ("value", "quality")) and len(with_fund) < 2 * n_quantiles:
            notes.append(
                f"value/quality skipped: cached fundamentals for {len(with_fund)} instrument(s) only"
            )
            wanted = [f for f in wanted if f not in ("value", "quality")]
        currencies = {i.currency.value for i in insts}
        if "size" in wanted and len(currencies) > 1:
            notes.append(
                f"size skipped: mixed currencies {sorted(currencies)} (market caps not comparable without FX)"
            )
            wanted = [f for f in wanted if f != "size"]
        if "size" in wanted and not with_fund:
            notes.append("size skipped: no shares outstanding in cached fundamentals")
            wanted = [f for f in wanted if f != "size"]
        sectors = {i.instrument_id: (i.sector or "unknown") for i in insts} if sector_neutral else None
        missing_px = [iid for iid in ids if iid not in closes.columns]
        if missing_px:
            notes.append(f"no price data for {missing_px}")

        def shares_at(as_of: date) -> dict[str, float]:
            out: dict[str, float] = {}
            for iid, s in stmts.items():
                v = s.as_of(as_of).ttm("shares_outstanding") or s.as_of(as_of).ttm("shares_diluted")
                if v:
                    out[iid] = float(v)
            return out

        def ratios_fn(iid: str, as_of: date):  # noqa: ANN202
            s = stmts.get(iid)
            if s is None or s.rows.empty or iid not in closes.columns:
                return None
            px = closes.loc[: pd.Timestamp(as_of), iid].dropna()
            return compute_ratios(s, price=float(px.iloc[-1]) if len(px) else None, as_of=as_of)

        def score_fn(t: pd.Timestamp) -> pd.DataFrame:
            as_of = t.date()
            raw = price_characteristics(closes, t, shares=shares_at(as_of) if "size" in wanted else None)
            if any(f in wanted for f in ("value", "quality")):
                raw = raw.join(fundamental_characteristics(with_fund, as_of, ratios_fn), how="left")
            norm = rank_normalise(raw.drop(columns=["price"]), groups=sectors)
            return factor_scores(norm, wanted)

        fr = factor_returns(
            closes,
            score_fn,
            factors=wanted,
            start=start_d,
            end=end_d,
            rebalance=rebalance,
            n_quantiles=n_quantiles,
        )
        fr.notes = notes + fr.notes
        return fr

    def factor_file(self, name: str) -> Path:
        return self.settings.paths.data_dir / "factors" / f"{name}.parquet"

    def save_factor_returns(self, name: str, returns: pd.DataFrame, notes: Sequence[str] = ()) -> Path:
        path = self.factor_file(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = returns.copy()
        out.index.name = "date"
        out.reset_index().to_parquet(path, index=False)
        path.with_suffix(".notes.txt").write_text("\n".join(notes), encoding="utf-8")
        return path

    def load_factor_returns(self, name: str) -> pd.DataFrame:
        path = self.factor_file(name)
        if not path.exists():
            raise ConfigError(f"no saved factor returns named {name!r} (looked in {path.parent})")
        df = pd.read_parquet(path)
        return df.set_index(pd.DatetimeIndex(pd.to_datetime(df["date"]))).drop(columns=["date"])

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
