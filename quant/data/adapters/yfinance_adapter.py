"""yfinance adapter: daily OHLCV plus dividends and splits, through one patchable network seam.

Facts that shape this adapter:
- Yahoo's `Close` is split-adjusted but NOT dividend-adjusted, so stored bars carry
  `AdjustmentState.SPLIT_ADJUSTED`. We pass `auto_adjust=False, actions=True` explicitly because yfinance's
  default flipped to auto-adjusting.
- LSE equities and many UK ETFs are quoted in pence (GBX). Prices and dividends are multiplied by
  `Instrument.price_scale`, so downstream code only ever sees pounds. Dividend units are a known soft spot
  (some UK payers declare in USD; Yahoo usually converts to GBp) and are sanity-checked in validation.
- Bars are indexed by the exchange-local date, which is the session date.
- The current session's bar is partial until the close; it is dropped by default
  (`drop_unfinished_session`), using the exchange calendar's real close (early closes included).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant.core.calendars import TradingCalendar, get_calendar
from quant.core.errors import AdapterError, CalendarError
from quant.core.time import now, to_date
from quant.core.types import BAR_COLUMNS, AdjustmentState, Instrument
from quant.data.adapters.base import (
    CORPORATE_ACTION_COLUMNS,
    AdapterResult,
    BaseAdapter,
    Capability,
    FieldMap,
    RateLimit,
    empty_bars,
)

HistoryFn = Callable[[str, date, date, str], pd.DataFrame]
FundamentalsFn = Callable[[str], Mapping[str, Any]]


def default_history(symbol: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """The network seam: exactly one yfinance call. Tests replace it with fixture-backed frames."""
    import yfinance as yf

    return yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
        interval=interval,
        auto_adjust=False,
        actions=True,
    )  # an unknown symbol comes back as an empty frame, which get_bars reports as a per-instrument error


def default_fundamentals(symbol: str) -> dict[str, Any]:
    """The second network seam: statements, earnings dates and profile for one symbol."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    return {
        "income": t.income_stmt,
        "balance": t.balance_sheet,
        "cashflow": t.cashflow,
        "income_q": t.quarterly_income_stmt,
        "balance_q": t.quarterly_balance_sheet,
        "cashflow_q": t.quarterly_cashflow,
        "earnings_dates": t.get_earnings_dates(limit=24),
        "info": t.info,
    }


def _empty_actions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CORPORATE_ACTION_COLUMNS})


class YFinanceAdapter(BaseAdapter):
    name = "yfinance"
    identifier_key = "yahoo"
    rate_limit = RateLimit(min_interval_s=0.2, note="unofficial API; keep the request rate modest")
    field_map = FieldMap(
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        adj_close="Adj Close",
        dividends="Dividends",
        splits="Stock Splits",
    )
    adjustment_state = AdjustmentState.SPLIT_ADJUSTED

    def __init__(
        self,
        history_fn: HistoryFn | None = None,
        drop_unfinished_session: bool = True,
        fundamentals_fn: FundamentalsFn | None = None,
    ) -> None:
        super().__init__()
        self._history: HistoryFn = history_fn or default_history
        self._fundamentals: FundamentalsFn = fundamentals_fn or default_fundamentals
        self.drop_unfinished_session = drop_unfinished_session

    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.BARS_DAILY,
                Capability.CORPORATE_ACTIONS,
                Capability.FX,
                Capability.FUNDAMENTALS,
                Capability.EARNINGS,
            }
        )

    # ------------------------------------------------------------------ fundamentals / earnings
    def get_fundamentals(self, instruments: Sequence[Instrument]) -> AdapterResult:
        from quant.fundamental.statements import from_yfinance

        frames, errors, info = [], {}, {}
        for inst in instruments:
            try:
                self._throttle()
                raw = self._fundamentals(self.symbol_for(inst))
                rows = from_yfinance(inst, raw)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if rows.empty:
                errors[inst.instrument_id] = "no statements returned"
                continue
            frames.append(rows)
            info[inst.instrument_id] = dict(raw.get("info") or {})
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"source": "yfinance statements; filing dates estimated"},
            extras={"info": info},
        )

    def get_earnings(self, instruments: Sequence[Instrument]) -> AdapterResult:
        frames, errors = [], {}
        for inst in instruments:
            try:
                self._throttle()
                raw = self._fundamentals(self.symbol_for(inst))
                ed = raw.get("earnings_dates")
                rows = self._parse_earnings(inst, ed)
            except Exception as exc:  # noqa: BLE001
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if rows.empty:
                errors[inst.instrument_id] = "no earnings dates returned"
                continue
            frames.append(rows)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return AdapterResult(
            vendor=self.name, frame=frame, errors=errors, requested=[i.instrument_id for i in instruments]
        )

    def _parse_earnings(self, inst: Instrument, ed: pd.DataFrame | None) -> pd.DataFrame:
        if ed is None or ed.empty:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(ed.index)
        dates = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        est = (
            pd.to_numeric(ed.get("EPS Estimate"), errors="coerce") if "EPS Estimate" in ed.columns else np.nan
        )
        act = (
            pd.to_numeric(ed.get("Reported EPS"), errors="coerce") if "Reported EPS" in ed.columns else np.nan
        )
        out = pd.DataFrame(
            {
                "instrument_id": inst.instrument_id,
                "report_date": dates,
                "eps_estimate": np.asarray(est, dtype="float64"),
                "eps_actual": np.asarray(act, dtype="float64"),
                "currency": inst.currency.value,
                "vendor": self.name,
                "recorded_at": pd.Timestamp(now()),
            }
        )
        out = out.drop_duplicates("report_date", keep="last").sort_values("report_date")
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ public API
    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        if freq != "1d":
            raise AdapterError(f"{self.name}: only daily bars are supported in this version (got {freq!r})")
        start_d, end_d = to_date(start), to_date(end)
        frames: list[pd.DataFrame] = []
        actions: list[pd.DataFrame] = []
        errors: dict[str, str] = {}
        for inst in instruments:
            symbol = self.symbol_for(inst)
            try:
                self._throttle()
                raw = self._history(symbol, start_d, end_d, freq)
                bars, acts = self._parse(inst, raw)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not sink the batch
                errors[inst.instrument_id] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            if bars.empty:
                errors[inst.instrument_id] = f"no data returned for {symbol!r} in {start_d}..{end_d}"
                continue
            frames.append(bars)
            if not acts.empty:
                actions.append(acts)
        frame = pd.concat(frames, ignore_index=True) if frames else empty_bars()
        extras = {"corporate_actions": pd.concat(actions, ignore_index=True) if actions else _empty_actions()}
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "freq": freq,
                "auto_adjust": False,
            },
            extras=extras,
        )

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult:
        res = self.get_bars(instruments, start, end)
        return AdapterResult(
            vendor=self.name,
            frame=res.extras["corporate_actions"],
            errors=res.errors,
            requested=res.requested,
            params=res.params,
        )

    # ------------------------------------------------------------------ parsing
    def _parse(self, inst: Instrument, raw: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if raw is None or raw.empty:
            return empty_bars(), _empty_actions()
        fm = self.field_map
        idx = pd.DatetimeIndex(raw.index)
        local_dates = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        scale = float(inst.price_scale)

        df = pd.DataFrame({"session_date": local_dates})
        for col, src in (("open", fm.open), ("high", fm.high), ("low", fm.low), ("close", fm.close)):
            df[col] = pd.to_numeric(raw[src], errors="coerce").to_numpy(dtype="float64") * scale
        volume = raw[fm.volume] if fm.volume in raw.columns else 0.0
        df["volume"] = pd.to_numeric(volume, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        df = df[df["close"].notna()].drop_duplicates("session_date", keep="last")
        df["instrument_id"] = inst.instrument_id
        df["currency"] = inst.currency.value
        df["vendor"] = self.name
        df["adjustment_state"] = self.adjustment_state.value

        cal = get_calendar(inst.calendar_code)
        df["ts_close_utc"] = self._closes_for(cal, df["session_date"])
        ingested_at = pd.Timestamp(now())
        df["ingested_at"] = ingested_at
        df = df.sort_values("session_date").reset_index(drop=True)

        if self.drop_unfinished_session and not df.empty:
            last_close = df["ts_close_utc"].iloc[-1]
            if pd.notna(last_close) and ingested_at < last_close:
                df = df.iloc[:-1]

        actions = self._parse_actions(inst, raw, local_dates, scale)
        return df.loc[:, list(BAR_COLUMNS)].reset_index(drop=True), actions

    @staticmethod
    def _closes_for(cal: TradingCalendar, session_dates: pd.Series) -> pd.Series:
        """Map each session date to the calendar's UTC close; NaT for dates the calendar does not know."""
        if session_dates.empty:
            return pd.Series(dtype="datetime64[ns, UTC]")
        start, end = session_dates.min().date(), session_dates.max().date()
        try:
            closes = cal.session_closes(start, end)
        except CalendarError:
            return pd.Series(pd.NaT, index=session_dates.index, dtype="datetime64[ns, UTC]")
        return session_dates.map(closes).astype("datetime64[ns, UTC]")

    def _parse_actions(
        self, inst: Instrument, raw: pd.DataFrame, local_dates: pd.DatetimeIndex, scale: float
    ) -> pd.DataFrame:
        fm = self.field_map
        rows: list[tuple[str, pd.Timestamp, str, float, str | None]] = []
        if fm.dividends and fm.dividends in raw.columns:
            div = pd.to_numeric(raw[fm.dividends], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            for d, v in zip(local_dates, div, strict=True):
                if v != 0.0:
                    rows.append((inst.instrument_id, d, "dividend", float(v) * scale, inst.currency.value))
        if fm.splits and fm.splits in raw.columns:
            spl = pd.to_numeric(raw[fm.splits], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
            for d, v in zip(local_dates, spl, strict=True):
                if v != 0.0:
                    rows.append((inst.instrument_id, d, "split", float(v), None))
        if not rows:
            return _empty_actions()
        out = pd.DataFrame(rows, columns=["instrument_id", "ex_date", "action_type", "value", "currency"])
        out["vendor"] = self.name
        out["recorded_at"] = pd.Timestamp(now())
        return out.loc[:, list(CORPORATE_ACTION_COLUMNS)]
