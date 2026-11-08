"""Ratio analysis from a point-in-time StatementSet plus market inputs. Every ratio is None when an input
is missing or its denominator is zero; nothing is silently filled."""

from __future__ import annotations

from datetime import date
from typing import Any

from quant.fundamental.statements import StatementSet

DEFAULT_TAX_RATE = 0.21  # used only when pretax income and tax are unavailable; reported in `assumptions`


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _interest_coverage(items: tuple[float | None, float | None]) -> float | None:
    ebit, interest = items
    return _div(ebit, abs(interest) if interest else None)


def compute_ratios(
    stmts: StatementSet,
    *,
    price: float | None = None,
    shares: float | None = None,
    as_of: date | str | None = None,
) -> dict[str, Any]:
    """Margins, returns, leverage, liquidity, efficiency, growth and (with price) market multiples."""
    s = stmts.as_of(as_of)
    t = s.ttm

    def pair(num: str, den: str) -> tuple[float | None, float | None]:
        """Two flow items on the same basis: TTM when both have four quarters, else both latest annual."""
        if s.quarters(num) >= 4 and s.quarters(den) >= 4:
            return t(num), t(den)
        return s.latest(num), s.latest(den)

    rev, op = t("revenue"), t("operating_income")
    ebit = t("ebit") if t("ebit") is not None else op
    ebitda, ni, ocf, fcf = t("ebitda"), t("net_income"), t("operating_cash_flow"), t("free_cash_flow")
    assets, equity, debt, cash = t("total_assets"), t("total_equity"), t("total_debt"), t("cash")
    cur_a, cur_l, divs = t("current_assets"), t("current_liabilities"), t("dividends_paid")
    shares_out = shares or t("shares_outstanding") or t("shares_diluted")
    assumptions: list[str] = []
    tax_rate = _div(*pair("income_tax", "pretax_income"))
    if tax_rate is None or not 0 <= tax_rate <= 0.5:
        tax_rate = DEFAULT_TAX_RATE
        assumptions.append(f"tax rate assumed {DEFAULT_TAX_RATE:.0%}")
    nopat = None if ebit is None else ebit * (1 - tax_rate)
    net_debt = None if debt is None else debt - (cash or 0.0)
    invested = None if equity is None else equity + (debt or 0.0) - (cash or 0.0)
    out: dict[str, Any] = {
        "as_of": None if as_of is None else str(as_of),
        "latest_annual_period": s.latest_period_end("annual"),
        "latest_quarter": s.latest_period_end("quarterly"),
        "filing_dates_estimated": s.filing_lag_estimated(),
        "revenue_ttm": rev,
        "net_income_ttm": ni,
        "ebitda_ttm": ebitda,
        "free_cash_flow_ttm": fcf,
        "gross_margin": _div(*pair("gross_profit", "revenue")),
        "operating_margin": _div(*pair("operating_income", "revenue")),
        "net_margin": _div(*pair("net_income", "revenue")),
        "fcf_margin": _div(*pair("free_cash_flow", "revenue")),
        "roe": _div(ni, equity),
        "roa": _div(ni, assets),
        "roic": _div(nopat, invested),
        "debt_to_equity": _div(debt, equity),
        "net_debt": net_debt,
        "net_debt_to_ebitda": _div(net_debt, ebitda),
        "interest_coverage": _interest_coverage(pair("ebit", "interest_expense")),
        "current_ratio": _div(cur_a, cur_l),
        "asset_turnover": _div(rev, assets),
        "accruals": _div(None if ni is None or ocf is None else ni - ocf, assets),
        "revenue_growth": s.yoy_growth("revenue"),
        "net_income_growth": s.yoy_growth("net_income"),
        "eps_growth": s.yoy_growth("eps_diluted"),
        "tax_rate": tax_rate,
        "shares": shares_out,
    }
    if price is not None and shares_out:
        mcap = price * shares_out
        ev = mcap + (net_debt or 0.0)
        eps = _div(ni, shares_out)
        out.update(
            {
                "price": price,
                "market_cap": mcap,
                "enterprise_value": ev,
                "pe": _div(price, eps) if eps and eps > 0 else None,
                "earnings_yield": _div(ni, mcap),
                "pb": _div(mcap, equity) if equity and equity > 0 else None,
                "ps": _div(mcap, rev),
                "ev_ebitda": _div(ev, ebitda) if ebitda and ebitda > 0 else None,
                "ev_sales": _div(ev, rev),
                "fcf_yield": _div(fcf, mcap),
                "dividend_yield": _div(abs(divs) if divs else None, mcap),
            }
        )
    out["assumptions"] = assumptions
    return out


RATIO_KEYS = (
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "roe",
    "roa",
    "roic",
    "debt_to_equity",
    "net_debt_to_ebitda",
    "interest_coverage",
    "current_ratio",
    "asset_turnover",
    "accruals",
    "revenue_growth",
    "net_income_growth",
    "eps_growth",
    "pe",
    "earnings_yield",
    "pb",
    "ps",
    "ev_ebitda",
    "ev_sales",
    "fcf_yield",
    "dividend_yield",
)
