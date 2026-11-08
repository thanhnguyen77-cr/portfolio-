"""Core domain types: instruments, enums and the canonical bar schema."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AssetClass(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FX = "fx"
    RATE = "rate"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FUND = "fund"


class Market(StrEnum):
    """Trading venue / calendar family. The value doubles as the calendar code."""

    XNYS = "XNYS"  # New York Stock Exchange (also used for NYSE Arca ETFs)
    XNAS = "XNAS"  # Nasdaq
    XLON = "XLON"  # London Stock Exchange
    CMES = "CMES"  # CME equity-index futures
    XCBT = "XCBT"  # CBOT grains / treasuries (Globex calendar: CMES)
    XCEC = "XCEC"  # COMEX metals (Globex calendar: CMES)
    XNYM = "XNYM"  # NYMEX energy, also a proxy for ICE Brent (Globex calendar: CMES)
    FX = "FX"  # spot FX: weekday proxy calendar, 17:00 New York close
    CRYPTO = "CRYPTO"  # 24/7

    @property
    def calendar_code(self) -> str:
        return _CALENDAR_FOR_MARKET.get(self, self.value)


# Venues without their own exchange_calendars entry borrow the calendar of the group they trade on.
_CALENDAR_FOR_MARKET: dict[Market, str] = {Market.XCBT: "CMES", Market.XCEC: "CMES", Market.XNYM: "CMES"}


class Currency(StrEnum):
    USD = "USD"
    GBP = "GBP"
    EUR = "EUR"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    NZD = "NZD"
    SEK = "SEK"
    NOK = "NOK"
    DKK = "DKK"
    HKD = "HKD"
    SGD = "SGD"
    CNY = "CNY"
    INR = "INR"
    ZAR = "ZAR"


class Frequency(StrEnum):
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    HOURLY = "1h"
    MIN5 = "5m"
    MIN1 = "1m"


class AdjustmentState(StrEnum):
    """How a stored price series has already been adjusted by its vendor.

    Yahoo's "Close" is split-adjusted but not dividend-adjusted; its "Adj Close" is both. The backtester
    fills on RAW or SPLIT_ADJUSTED prices only, and adjusts quantities for splits only when RAW.
    """

    RAW = "raw"
    SPLIT_ADJUSTED = "split_adjusted"
    FULLY_ADJUSTED = "fully_adjusted"


class StampDutyClass(StrEnum):
    """Drives UK/Irish transaction taxes. UNKNOWN is treated conservatively (tax applied)."""

    UK_ORDINARY = "uk_ordinary"  # UK-incorporated ordinary shares incl. investment trusts: SDRT 0.5 % on buys
    IRISH = "irish"  # Irish-incorporated: 1 % stamp duty on buys
    AIM = "aim"  # AIM-quoted growth-market shares: exempt
    EXEMPT = "exempt"  # ETFs, non-UK incorporated, depositary receipts etc.
    UNKNOWN = "unknown"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


_SLUG_RE = re.compile(r"[^A-Z0-9]+")


def slugify_symbol(symbol: str) -> str:
    """Upper-case, filesystem-safe slug: 'BRK-B' -> 'BRK_B', '^GSPC' -> 'GSPC'."""
    slug = _SLUG_RE.sub("_", str(symbol).strip().upper()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a slug from symbol {symbol!r}")
    return slug


def make_instrument_id(market: Market | str, symbol: str) -> str:
    """Surrogate key `<MARKET>_<SLUG>`, e.g. XLON_VOD, XNYS_BRK_B, FX_GBPUSD."""
    return f"{Market(market).value}_{slugify_symbol(symbol)}"


class Instrument(BaseModel):
    """A tradable or observable instrument. Immutable; identified by `instrument_id`, never by ticker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument_id: str = Field(default="", description="Surrogate key <MARKET>_<SLUG>; derived when omitted.")
    symbol: str = Field(description="Canonical vendor-neutral symbol: VOD (not VOD.L), GBPUSD, SPX.")
    asset_class: AssetClass
    market: Market
    currency: Currency = Field(description="Quote currency AFTER price_scale is applied (GBP, never GBX).")
    name: str | None = None
    price_scale: float = Field(default=1.0, description="Vendor prices are multiplied by this; 0.01 for GBX.")
    stamp_duty_class: StampDutyClass = StampDutyClass.UNKNOWN
    tradable: bool = True
    sector: str | None = None
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2 of incorporation.")
    lot_size: int = 1
    identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="vendor -> vendor symbol (yahoo, stooq, fred, boe, bbg, ric, isin, sedol, figi, alpaca)",
    )
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data: Any) -> Any:
        needs_id = isinstance(data, dict) and not data.get("instrument_id")
        if needs_id and data.get("market") and data.get("symbol"):
            data = dict(data)
            data["instrument_id"] = make_instrument_id(data["market"], data["symbol"])
        return data

    @field_validator("price_scale")
    @classmethod
    def _positive_scale(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price_scale must be > 0")
        return v

    @field_validator("lot_size")
    @classmethod
    def _positive_lot(cls, v: int) -> int:
        if v < 1:
            raise ValueError("lot_size must be >= 1")
        return v

    @property
    def calendar_code(self) -> str:
        return self.market.calendar_code

    @property
    def quoted_in_pence(self) -> bool:
        return self.currency == Currency.GBP and self.price_scale == 0.01

    def vendor_symbol(self, vendor: str) -> str:
        """The symbol a vendor knows this instrument by; falls back to the canonical symbol."""
        return self.identifiers.get(vendor, self.symbol)

    def scale_price(self, vendor_price: float) -> float:
        """Convert a vendor-quoted price into `currency` units (e.g. pence -> pounds)."""
        return vendor_price * self.price_scale


BAR_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "currency",
    "vendor",
    "adjustment_state",
    "ts_close_utc",
    "ingested_at",
)


def bar_schema() -> Any:
    """The canonical pyarrow schema for daily bars (one row per instrument x session_date)."""
    import pyarrow as pa

    return pa.schema(
        [
            ("instrument_id", pa.string()),
            ("session_date", pa.date32()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
            ("currency", pa.string()),
            ("vendor", pa.string()),
            ("adjustment_state", pa.string()),
            ("ts_close_utc", pa.timestamp("ns", tz="UTC")),
            ("ingested_at", pa.timestamp("ns", tz="UTC")),
        ]
    )
