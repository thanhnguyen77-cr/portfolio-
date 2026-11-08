"""Macro analysis: series catalogue, yield curve, inflation transforms and a cross-asset regime board."""

from quant.macro.catalog import MACRO_SERIES, MacroSeries, macro_series_spec
from quant.macro.inflation import annualised_3m, breakeven, inflation_regime, mom, real_rate, yoy
from quant.macro.regime import cross_asset_dashboard, growth_inflation_quadrant
from quant.macro.verify import summarize as summarize_verification
from quant.macro.verify import verify_catalog
from quant.macro.yield_curve import curvature, curve_snapshot, inversion_flags, slope

__all__ = [
    "MACRO_SERIES",
    "MacroSeries",
    "annualised_3m",
    "breakeven",
    "cross_asset_dashboard",
    "curvature",
    "curve_snapshot",
    "growth_inflation_quadrant",
    "inflation_regime",
    "inversion_flags",
    "macro_series_spec",
    "mom",
    "real_rate",
    "slope",
    "summarize_verification",
    "verify_catalog",
    "yoy",
]
