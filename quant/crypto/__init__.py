"""Crypto (optional): 24/7 calendar handling for pairs in the instrument catalogue.

Crypto pairs trade every calendar day, so their daily series have ~365 observations a year and their
weekend moves have no counterpart in equity or FX data. This module makes those facts explicit instead of
letting them leak into cross-asset analytics: an annualisation factor of 365, an alignment helper that
folds weekend returns into the next session of an exchange calendar, and the caveats every crypto result
should carry (venue fragmentation, composite index prices from Yahoo, no central close).
"""

from quant.crypto.pairs import (
    CRYPTO_CAVEATS,
    CRYPTO_PERIODS_PER_YEAR,
    align_to_calendar,
    crypto_pairs,
    realised_vol_crypto,
    weekend_share_of_variance,
)

__all__ = [
    "CRYPTO_CAVEATS",
    "CRYPTO_PERIODS_PER_YEAR",
    "align_to_calendar",
    "crypto_pairs",
    "realised_vol_crypto",
    "weekend_share_of_variance",
]
