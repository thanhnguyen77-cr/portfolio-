"""Free-cash-flow-to-firm DCF with explicit assumptions and a sensitivity grid. A valuation is only as good
as its inputs; the result object carries every assumption so a report can show them next to the number."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DCFResult:
    enterprise_value: float
    equity_value: float
    per_share: float | None
    pv_explicit: float
    pv_terminal: float
    terminal_share: float
    cash_flows: list[float]
    discount_rate: float
    terminal_growth: float
    net_debt: float
    shares: float | None
    assumptions: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "per_share": self.per_share,
            "pv_explicit": self.pv_explicit,
            "pv_terminal": self.pv_terminal,
            "terminal_share": self.terminal_share,
            "discount_rate": self.discount_rate,
            "terminal_growth": self.terminal_growth,
            "net_debt": self.net_debt,
            "shares": self.shares,
            **{f"assumption_{k}": v for k, v in self.assumptions.items()},
        }


def wacc(
    rf: float, erp: float, beta: float, cost_of_debt: float, tax_rate: float, debt_weight: float
) -> float:
    """Weighted average cost of capital: E/V (rf + beta x ERP) + D/V kd (1 - t)."""
    if not 0 <= debt_weight <= 1:
        raise ValueError("debt_weight must be within [0, 1]")
    cost_of_equity = rf + beta * erp
    return (1 - debt_weight) * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)


def dcf_valuation(
    fcf0: float,
    growth: float | Sequence[float],
    *,
    years: int = 5,
    terminal_growth: float = 0.02,
    discount_rate: float = 0.08,
    net_debt: float = 0.0,
    shares: float | None = None,
    mid_year: bool = False,
) -> DCFResult:
    """Project FCF from `fcf0` at `growth` (scalar or per-year path), discount, add a Gordon terminal."""
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must exceed terminal_growth")
    path = [float(growth)] * years if isinstance(growth, int | float) else [float(g) for g in growth]
    if not path:
        raise ValueError("growth path must have at least one year")
    flows: list[float] = []
    level = float(fcf0)
    for g in path:
        level *= 1.0 + g
        flows.append(level)
    n = len(flows)
    t = np.arange(1, n + 1, dtype="float64") - (0.5 if mid_year else 0.0)
    factors = (1.0 + discount_rate) ** t
    pv_explicit = float(np.sum(np.asarray(flows) / factors))
    terminal = flows[-1] * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = float(terminal / (1.0 + discount_rate) ** (n - (0.5 if mid_year else 0.0)))
    ev = pv_explicit + pv_terminal
    equity = ev - net_debt
    per_share = equity / shares if shares else None
    return DCFResult(
        ev,
        equity,
        per_share,
        pv_explicit,
        pv_terminal,
        pv_terminal / ev if ev else float("nan"),
        flows,
        discount_rate,
        terminal_growth,
        net_debt,
        shares,
        {"fcf0": fcf0, "growth_path": path, "years": n, "mid_year": mid_year},
    )


def dcf_sensitivity(
    fcf0: float,
    growth: float | Sequence[float],
    *,
    discount_rates: Sequence[float],
    terminal_growths: Sequence[float],
    years: int = 5,
    net_debt: float = 0.0,
    shares: float | None = None,
) -> pd.DataFrame:
    """Per-share value (equity value when shares is None) over a discount-rate x terminal-growth grid."""
    table = pd.DataFrame(
        index=[f"{r:.1%}" for r in discount_rates],
        columns=[f"{g:.1%}" for g in terminal_growths],
        dtype="float64",
    )
    table.index.name = "discount_rate"
    table.columns.name = "terminal_growth"
    for r, r_label in zip(discount_rates, table.index, strict=True):
        for g, g_label in zip(terminal_growths, table.columns, strict=True):
            if r <= g:
                table.loc[r_label, g_label] = np.nan
                continue
            res = dcf_valuation(
                fcf0,
                growth,
                years=years,
                terminal_growth=g,
                discount_rate=r,
                net_debt=net_debt,
                shares=shares,
            )
            table.loc[r_label, g_label] = res.per_share if shares else res.equity_value
    return table
