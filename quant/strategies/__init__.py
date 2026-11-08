"""Reference strategies. They exist to exercise and document the engine, not as recommendations."""

from quant.strategies.buy_and_hold import BuyAndHold
from quant.strategies.trend_following import TrendFollowing

__all__ = ["BuyAndHold", "TrendFollowing"]
