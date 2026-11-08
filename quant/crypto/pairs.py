from __future__ import annotations

import numpy as np
import pandas as pd

from quant.core.calendars import get_calendar
from quant.core.catalog import InstrumentCatalog
from quant.core.types import AssetClass, Instrument

CRYPTO_PERIODS_PER_YEAR = 365

CRYPTO_CAVEATS = [
    "Crypto trades 24/7: 'daily' bars are UTC calendar days with no exchange close; annualise with 365.",
    "Yahoo's BTC-USD / ETH-USD are composite index prices across venues, not a tradable quote on any one of "
    "them; fills, fees and spreads differ by venue.",
    "Weekend returns have no equity or FX counterpart; cross-asset correlations need `align_to_calendar`.",
    "No stamp duty or SEC fees apply, but venue fees (maker/taker) are typically far above equity costs.",
]


def crypto_pairs(catalog: InstrumentCatalog) -> list[Instrument]:
    return catalog.filter(asset_class=AssetClass.CRYPTO)


def realised_vol_crypto(prices: pd.Series, window: int = 30) -> pd.Series:
    """Rolling realised volatility annualised with 365 calendar days."""
    rets = np.log(pd.Series(prices, dtype=float)).diff()
    return (rets.rolling(window).std(ddof=1) * np.sqrt(CRYPTO_PERIODS_PER_YEAR)).rename("realised_vol")


def align_to_calendar(prices: pd.Series, calendar_code: str = "XNYS") -> pd.Series:
    """Resample a 24/7 price series onto an exchange calendar: each session takes the last price at or
    before that date, so returns on Mondays (and days after holidays) absorb the intervening moves."""
    s = pd.Series(prices, dtype=float).dropna().sort_index()
    if s.empty:
        return s
    idx = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = pd.Series(s.to_numpy(), index=idx)
    s = s[~s.index.duplicated(keep="last")]
    cal = get_calendar(calendar_code)
    sessions = cal.sessions(idx[0].date(), idx[-1].date())
    out = s.reindex(s.index.union(sessions)).ffill().reindex(sessions)
    return out.dropna().rename(f"{getattr(prices, 'name', 'price')}@{calendar_code}")


def weekend_share_of_variance(prices: pd.Series) -> float:
    """Share of total daily log-return variance that occurs on Saturdays and Sundays (UTC)."""
    s = pd.Series(prices, dtype=float).dropna()
    rets = np.log(s).diff().dropna()
    if rets.empty:
        return float("nan")
    weekend = pd.DatetimeIndex(rets.index).day_of_week >= 5
    total = float((rets**2).sum())
    return float((rets[weekend] ** 2).sum() / total) if total > 0 else float("nan")
