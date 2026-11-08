"""Vendor adapters. Every adapter implements `quant.data.adapters.base.MarketDataAdapter`.

Free / keyed: yfinance, stooq, fred, boe, ons, ndl, sec_edgar, alpaca_data, polygon, tiingo, companies_house.
Licensed stubs (contract + wiring guidance only): bloomberg, refinitiv, factset, capiq.
"""

from quant.data.adapters.base import (
    AdapterResult,
    BaseAdapter,
    BaseHTTPAdapter,
    Capability,
    FieldMap,
    MarketDataAdapter,
    RateLimit,
)

__all__ = [
    "AdapterResult",
    "BaseAdapter",
    "BaseHTTPAdapter",
    "Capability",
    "FieldMap",
    "MarketDataAdapter",
    "RateLimit",
]
