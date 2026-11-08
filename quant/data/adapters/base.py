"""The adapter contract shared by every data vendor, free or licensed.

Design points (so a Bloomberg/Refinitiv feed can drop in later):
- Methods are **batched** over instruments and return an `AdapterResult` whose `errors` map holds
  per-instrument failures (entitlement, unknown symbol, ...) instead of raising on the first one.
- Instruments carry vendor identifiers (`Instrument.identifiers[vendor]`), so adapters never guess tickers.
- Adapters are context managers (session lifecycle) and declare a `RateLimit` and a `FieldMap`.
- Network access goes through one seam (`BaseHTTPAdapter._http`, or a vendor SDK wrapper) that tests patch.
"""

from __future__ import annotations

import time as _time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import requests
from tenacity import RetryError, Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from quant.core.errors import AdapterError
from quant.core.types import BAR_COLUMNS, Instrument


class Capability(StrEnum):
    BARS_DAILY = "bars_daily"
    BARS_INTRADAY = "bars_intraday"
    CORPORATE_ACTIONS = "corporate_actions"
    FUNDAMENTALS = "fundamentals"
    EARNINGS = "earnings"
    FX = "fx"
    MACRO = "macro"
    FILINGS = "filings"
    CONSTITUENTS = "constituents"


@dataclass(frozen=True)
class RateLimit:
    """Politeness settings. `min_interval_s` is enforced by `BaseAdapter._throttle()`."""

    min_interval_s: float = 0.0
    requests_per_minute: int | None = None
    note: str = ""


@dataclass(frozen=True)
class FieldMap:
    """Vendor column names for the canonical OHLCV and corporate-action fields (None = not provided)."""

    open: str = "Open"
    high: str = "High"
    low: str = "Low"
    close: str = "Close"
    volume: str = "Volume"
    adj_close: str | None = "Adj Close"
    dividends: str | None = "Dividends"
    splits: str | None = "Stock Splits"


CORPORATE_ACTION_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "ex_date",
    "action_type",  # 'dividend' | 'split'
    "value",  # cash amount per share (dividend) or ratio new/old (split)
    "currency",  # dividend currency; None for splits
    "vendor",
    "recorded_at",
)

MACRO_COLUMNS: tuple[str, ...] = ("series_id", "date", "value", "vendor", "ingested_at")


@dataclass
class AdapterResult:
    """A batched adapter call: the rows that worked plus a per-instrument map of what failed."""

    vendor: str
    frame: pd.DataFrame
    errors: dict[str, str] = field(default_factory=dict)
    requested: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def ok(self) -> list[str]:
        if self.frame.empty or "instrument_id" not in self.frame.columns:
            return []
        return sorted(self.frame["instrument_id"].astype(str).unique().tolist())

    @property
    def failed(self) -> list[str]:
        return sorted(self.errors)

    @property
    def all_failed(self) -> bool:
        return bool(self.requested) and self.frame.empty and len(self.errors) >= len(self.requested)

    def raise_if_all_failed(self) -> AdapterResult:
        if self.all_failed:
            detail = "; ".join(f"{k}: {v}" for k, v in sorted(self.errors.items()))
            raise AdapterError(f"{self.vendor}: every requested instrument failed ({detail})")
        return self

    def summary(self) -> str:
        return (
            f"{self.vendor}: {len(self.frame)} rows for {len(self.ok)} instrument(s); "
            f"{len(self.errors)} failed out of {len(self.requested)} requested"
        )


def empty_bars() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in BAR_COLUMNS})


@runtime_checkable
class MarketDataAdapter(Protocol):
    """Structural interface every adapter satisfies (checked by the contract tests)."""

    name: str
    rate_limit: RateLimit
    field_map: FieldMap

    def capabilities(self) -> frozenset[Capability]: ...

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult: ...

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult: ...

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult: ...

    def get_fundamentals(self, instruments: Sequence[Instrument]) -> AdapterResult: ...

    def get_earnings(self, instruments: Sequence[Instrument]) -> AdapterResult: ...

    def __enter__(self) -> MarketDataAdapter: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class BaseAdapter:
    """Default behaviour: declares nothing, rejects everything, throttles politely."""

    name: str = "base"
    identifier_key: str | None = None  # key into Instrument.identifiers; defaults to `name`
    rate_limit: RateLimit = RateLimit()
    field_map: FieldMap = FieldMap()

    def __init__(self) -> None:
        self._last_call: float = 0.0

    def symbol_for(self, inst: Instrument) -> str:
        """Vendor symbol for `inst` (e.g. VOD.L for Yahoo), or the canonical symbol when none is mapped."""
        return inst.vendor_symbol(self.identifier_key or self.name)

    def capabilities(self) -> frozenset[Capability]:
        return frozenset()

    def supports(self, capability: Capability | str) -> bool:
        return Capability(capability) in self.capabilities()

    def _unsupported(self, capability: Capability) -> AdapterError:
        return AdapterError(f"adapter {self.name!r} does not provide {capability.value}")

    def _throttle(self) -> None:
        interval = self.rate_limit.min_interval_s
        if interval <= 0:
            return
        elapsed = _time.monotonic() - self._last_call
        if elapsed < interval:
            _time.sleep(interval - elapsed)
        self._last_call = _time.monotonic()

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        raise self._unsupported(Capability.BARS_DAILY)

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult:
        raise self._unsupported(Capability.CORPORATE_ACTIONS)

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        raise self._unsupported(Capability.MACRO)

    def get_fundamentals(self, instruments: Sequence[Instrument]) -> AdapterResult:
        raise self._unsupported(Capability.FUNDAMENTALS)

    def get_earnings(self, instruments: Sequence[Instrument]) -> AdapterResult:
        raise self._unsupported(Capability.EARNINGS)

    def __enter__(self) -> BaseAdapter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        return None


class BaseHTTPAdapter(BaseAdapter):
    """HTTP vendors: one retrying `_http()` seam that tests replace with fixture-backed fakes."""

    user_agent: str = "quant-platform/0.1 (+https://github.com/thanhnguyen77-cr/portfolio-)"
    timeout_s: float = 30.0
    max_attempts: int = 3
    retry_min_wait_s: float = 0.5
    retry_max_wait_s: float = 8.0

    def __init__(self, session: requests.Session | None = None) -> None:
        super().__init__()
        self._session = session or requests.Session()
        self._owns_session = session is None

    def _http(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        data: Any = None,
        auth: Any = None,
    ) -> requests.Response:
        """GET/POST with exponential-backoff retries on connection errors and 5xx. 4xx raise immediately."""
        hdrs = {"User-Agent": self.user_agent, **(headers or {})}

        def _once() -> requests.Response:
            self._throttle()
            resp = self._session.request(
                method, url, params=params, headers=hdrs, data=data, auth=auth, timeout=self.timeout_s
            )
            if 500 <= resp.status_code < 600:
                raise requests.ConnectionError(f"{resp.status_code} from {url}")
            if resp.status_code >= 400:
                raise AdapterError(f"{self.name}: HTTP {resp.status_code} for {url}: {resp.text[:200]}")
            return resp

        retrying = Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=0.5, min=self.retry_min_wait_s, max=self.retry_max_wait_s),
            retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
            reraise=False,
        )
        try:
            return retrying(_once)
        except RetryError as exc:
            msg = f"{self.name}: request to {url} failed after {self.max_attempts} attempts"
            raise AdapterError(msg) from exc

    def close(self) -> None:
        if self._owns_session:
            self._session.close()
