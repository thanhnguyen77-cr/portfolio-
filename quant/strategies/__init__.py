"""Reference strategies. They exist to exercise and document the engine, not as recommendations."""

from quant.strategies.buy_and_hold import BuyAndHold
from quant.strategies.factor_tilt import FactorTilt
from quant.strategies.fixed_weights import FixedWeights
from quant.strategies.macro_regime_switch import MacroRegimeSwitch
from quant.strategies.pairs_cointegration import PairsCointegration
from quant.strategies.trend_following import TrendFollowing
from quant.strategies.xs_momentum import CrossSectionalMomentum

__all__ = [
    "BuyAndHold",
    "CrossSectionalMomentum",
    "FactorTilt",
    "FixedWeights",
    "MacroRegimeSwitch",
    "PairsCointegration",
    "TrendFollowing",
]
