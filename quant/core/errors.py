"""Exception hierarchy for the quant package."""

from __future__ import annotations


class QuantError(Exception):
    """Base class for all quant errors."""


class ConfigError(QuantError):
    """Invalid or missing configuration."""


class CalendarError(QuantError):
    """Unknown calendar, or a date outside the calendar's bounds."""


class AdapterError(QuantError):
    """A data adapter failed."""


class AdapterNotConfigured(AdapterError):
    """The adapter exists but is missing credentials or a licensed SDK."""


class DataQualityError(QuantError):
    """Data failed validation badly enough to stop."""


class StoreError(QuantError):
    """Parquet/DuckDB storage failure."""


class UniverseError(QuantError):
    """Universe construction or membership failure."""


class BacktestError(QuantError):
    """Backtest configuration or execution failure."""


class BrokerError(QuantError):
    """Paper broker failure."""


class FXError(QuantError):
    """Missing or unusable FX rate."""


class LiveTradingDisabled(QuantError):
    """Raised whenever code reaches a live-trading path. Live trading is disabled by design."""
