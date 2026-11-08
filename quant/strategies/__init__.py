"""Reference strategies. They exist to exercise and document the engine, not as recommendations."""

from quant.strategies.buy_and_hold import BuyAndHold
from quant.strategies.trend_following import TrendFollowing
from quant.strategies.xs_momentum import CrossSectionalMomentum

__all__ = ["BuyAndHold", "CrossSectionalMomentum", "TrendFollowing"]
