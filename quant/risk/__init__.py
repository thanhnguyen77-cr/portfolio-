"""Risk analytics: VaR/CVaR family, stress windows, factor scenarios, correlation structure, drawdowns."""

from quant.risk.correlation import (
    average_correlation,
    correlation_breakdown,
    correlation_summary,
    diversification_ratio,
    rolling_absorption,
)
from quant.risk.drawdown import (
    calmar_ratio,
    conditional_drawdown_at_risk,
    drawdown_episodes,
    pain_index,
    ulcer_index,
    underwater_summary,
)
from quant.risk.report import build_risk_report, portfolio_returns
from quant.risk.scenario import estimate_betas, factor_scenario, scenario_table
from quant.risk.stress import STRESS_WINDOWS, StressWindow, beta_shock, historical_stress, hypothetical_shocks
from quant.risk.var import (
    VaRResult,
    cornish_fisher_var,
    cornish_fisher_z,
    filtered_historical_var,
    historical_var,
    kupiec_test,
    monte_carlo_var,
    parametric_var,
    rolling_historical_var,
    var_table,
)

__all__ = [
    "STRESS_WINDOWS",
    "StressWindow",
    "VaRResult",
    "average_correlation",
    "beta_shock",
    "build_risk_report",
    "calmar_ratio",
    "conditional_drawdown_at_risk",
    "cornish_fisher_var",
    "cornish_fisher_z",
    "correlation_breakdown",
    "correlation_summary",
    "diversification_ratio",
    "drawdown_episodes",
    "estimate_betas",
    "factor_scenario",
    "filtered_historical_var",
    "historical_stress",
    "historical_var",
    "hypothetical_shocks",
    "kupiec_test",
    "monte_carlo_var",
    "pain_index",
    "parametric_var",
    "portfolio_returns",
    "rolling_absorption",
    "rolling_historical_var",
    "scenario_table",
    "ulcer_index",
    "underwater_summary",
    "var_table",
]
