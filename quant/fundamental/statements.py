"""Point-in-time financial statements.

Every row carries both `period_end` (the fiscal period it describes) and `filed_at` (when the market could
have known it). `StatementSet.as_of(date)` only sees rows with `filed_at <= date`, so a backtest cannot
use a quarter before it was reported. SEC EDGAR gives true filing dates; vendors that only give period
ends get an estimated filing date (period end + a disclosure lag) flagged `filed_at_estimated=True`.

Canonical items (long format) are listed in `ITEMS`; vendor line names map onto them via the tables below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant.core.time import now, to_date
from quant.core.types import Instrument

FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "instrument_id",
    "statement",
    "item",
    "period_end",
    "period_type",
    "value",
    "currency",
    "filed_at",
    "filed_at_estimated",
    "vendor",
    "recorded_at",
)

ITEMS: dict[str, str] = {  # canonical item -> statement
    "revenue": "income",
    "cost_of_revenue": "income",
    "gross_profit": "income",
    "operating_income": "income",
    "ebit": "income",
    "ebitda": "income",
    "interest_expense": "income",
    "pretax_income": "income",
    "income_tax": "income",
    "net_income": "income",
    "eps_diluted": "income",
    "shares_diluted": "income",
    "depreciation_amortization": "cashflow",
    "operating_cash_flow": "cashflow",
    "capex": "cashflow",
    "free_cash_flow": "cashflow",
    "dividends_paid": "cashflow",
    "total_assets": "balance",
    "current_assets": "balance",
    "cash": "balance",
    "receivables": "balance",
    "inventory": "balance",
    "current_liabilities": "balance",
    "total_liabilities": "balance",
    "total_debt": "balance",
    "long_term_debt": "balance",
    "short_term_debt": "balance",
    "total_equity": "balance",
    "shares_outstanding": "balance",
}

FLOW_ITEMS = frozenset(item for item, stmt in ITEMS.items() if stmt in {"income", "cashflow"})

# yfinance line names (first match wins) -> canonical item
YFINANCE_MAP: dict[str, tuple[str, ...]] = {
    "revenue": ("Total Revenue", "Operating Revenue"),
    "cost_of_revenue": ("Cost Of Revenue", "Reconciled Cost Of Revenue"),
    "gross_profit": ("Gross Profit",),
    "operating_income": ("Operating Income", "Total Operating Income As Reported"),
    "ebit": ("EBIT",),
    "ebitda": ("EBITDA", "Normalized EBITDA"),
    "interest_expense": ("Interest Expense", "Interest Expense Non Operating"),
    "pretax_income": ("Pretax Income",),
    "income_tax": ("Tax Provision",),
    "net_income": ("Net Income", "Net Income Common Stockholders"),
    "eps_diluted": ("Diluted EPS",),
    "shares_diluted": ("Diluted Average Shares",),
    "depreciation_amortization": ("Depreciation And Amortization", "Depreciation Amortization Depletion"),
    "operating_cash_flow": ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
    "capex": ("Capital Expenditure",),
    "free_cash_flow": ("Free Cash Flow",),
    "dividends_paid": ("Cash Dividends Paid", "Common Stock Dividend Paid"),
    "total_assets": ("Total Assets",),
    "current_assets": ("Current Assets",),
    "cash": ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
    "receivables": ("Accounts Receivable", "Receivables"),
    "inventory": ("Inventory",),
    "current_liabilities": ("Current Liabilities",),
    "total_liabilities": ("Total Liabilities Net Minority Interest",),
    "total_debt": ("Total Debt",),
    "long_term_debt": ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"),
    "short_term_debt": ("Current Debt", "Current Debt And Capital Lease Obligation"),
    "total_equity": ("Stockholders Equity", "Total Equity Gross Minority Interest"),
    "shares_outstanding": ("Ordinary Shares Number", "Share Issued"),
}

# us-gaap concepts (first available wins) -> canonical item
EDGAR_MAP: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold"),
    "gross_profit": ("GrossProfit",),
    "operating_income": ("OperatingIncomeLoss",),
    "interest_expense": ("InterestExpense",),
    "pretax_income": (
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ),
    "income_tax": ("IncomeTaxExpenseBenefit",),
    "net_income": ("NetIncomeLoss",),
    "eps_diluted": ("EarningsPerShareDiluted",),
    "shares_diluted": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    "depreciation_amortization": ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization"),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "dividends_paid": ("PaymentsOfDividends", "PaymentsOfDividendsCommonStock"),
    "total_assets": ("Assets",),
    "current_assets": ("AssetsCurrent",),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",),
    "receivables": ("AccountsReceivableNetCurrent",),
    "inventory": ("InventoryNet",),
    "current_liabilities": ("LiabilitiesCurrent",),
    "total_liabilities": ("Liabilities",),
    "long_term_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "short_term_debt": ("LongTermDebtCurrent", "DebtCurrent"),
    "total_equity": ("StockholdersEquity",),
    "shares_outstanding": ("CommonStockSharesOutstanding",),
}

DEFAULT_LAG_DAYS = {"annual": 75, "quarterly": 45}


def _empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in FUNDAMENTAL_COLUMNS})


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty()
    df = pd.DataFrame(rows)
    df["period_end"] = pd.to_datetime(df["period_end"]).dt.date
    df["filed_at"] = pd.to_datetime(df["filed_at"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df["recorded_at"] = pd.Timestamp(now())
    df = df.dropna(subset=["value"])
    df = df.sort_values(["period_end", "filed_at"]).drop_duplicates(
        ["item", "period_end", "period_type"], keep="last"
    )
    return df.loc[:, list(FUNDAMENTAL_COLUMNS)].reset_index(drop=True)


# ----------------------------------------------------------------------------- yfinance


def from_yfinance(
    inst: Instrument,
    frames: Mapping[str, pd.DataFrame | None],
    *,
    lag_days: Mapping[str, int] | None = None,
    vendor: str = "yfinance",
) -> pd.DataFrame:
    """frames: income/balance/cashflow (annual) and income_q/balance_q/cashflow_q (quarterly), yfinance layout
    (rows = line items, columns = period-end timestamps). Filing dates are estimated: period end + lag."""
    lags = {**DEFAULT_LAG_DAYS, **(lag_days or {})}
    rows: list[dict[str, Any]] = []
    for key, period_type in (
        ("income", "annual"),
        ("balance", "annual"),
        ("cashflow", "annual"),
        ("income_q", "quarterly"),
        ("balance_q", "quarterly"),
        ("cashflow_q", "quarterly"),
    ):
        frame = frames.get(key)
        if frame is None or frame.empty:
            continue
        statement = key.replace("_q", "")
        for item, names in YFINANCE_MAP.items():
            if ITEMS[item] != statement:
                continue
            line = next((n for n in names if n in frame.index), None)
            if line is None:
                continue
            for col in frame.columns:
                value = frame.loc[line, col]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                period_end = to_date(pd.Timestamp(col))
                rows.append(
                    {
                        "instrument_id": inst.instrument_id,
                        "statement": statement,
                        "item": item,
                        "period_end": period_end,
                        "period_type": period_type,
                        "value": float(value),
                        "currency": inst.currency.value,
                        "filed_at": period_end + timedelta(days=lags[period_type]),
                        "filed_at_estimated": True,
                        "vendor": vendor,
                    }
                )
    out = _rows_to_frame(rows)
    return _derive(out, inst, vendor)


# ----------------------------------------------------------------------------- EDGAR


def from_edgar_companyfacts(
    inst: Instrument, facts: Mapping[str, Any], *, vendor: str = "sec_edgar"
) -> pd.DataFrame:
    """SEC `companyfacts` JSON -> canonical rows with true `filed` dates (10-K annual, 10-Q quarterly).

    Flow items in 10-K facts are full-year values; quarterly flow values are only kept when the fact's
    `frame` marks a single quarter (CYyyyyQn), since cumulative year-to-date values would otherwise leak in.
    """
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    currency = inst.currency.value
    rows: list[dict[str, Any]] = []
    for item, concepts in EDGAR_MAP.items():
        concept = next((c for c in concepts if c in gaap), None)
        if concept is None:
            continue
        units = gaap[concept].get("units") or {}
        unit_key = next(
            (
                u
                for u in (currency, "USD", "shares", f"{currency}/shares", "USD/shares", "pure")
                if u in units
            ),
            None,
        )
        if unit_key is None:
            continue
        for fact in units[unit_key]:
            form = str(fact.get("form", ""))
            fp = str(fact.get("fp", ""))
            frame = str(fact.get("frame", "") or "")
            if form.startswith("10-K") or fp == "FY":
                period_type = "annual"
                if item in FLOW_ITEMS and frame and "Q" in frame:
                    continue  # a quarterly frame reported inside a 10-K: not the annual figure
            elif form.startswith("10-Q"):
                period_type = "quarterly"
                if item in FLOW_ITEMS and not (frame and frame.endswith(("Q1", "Q2", "Q3", "Q4"))):
                    continue  # year-to-date flow in a 10-Q: skip, keep single-quarter frames only
            else:
                continue
            rows.append(
                {
                    "instrument_id": inst.instrument_id,
                    "statement": ITEMS[item],
                    "item": item,
                    "period_end": to_date(fact["end"]),
                    "period_type": period_type,
                    "value": float(fact["val"]),
                    "currency": currency if unit_key in (currency, "USD") else None,
                    "filed_at": to_date(fact["filed"]),
                    "filed_at_estimated": False,
                    "vendor": vendor,
                }
            )
    out = _rows_to_frame(rows)
    return _derive(out, inst, vendor)


# ----------------------------------------------------------------------------- derived items


def _derive(df: pd.DataFrame, inst: Instrument, vendor: str) -> pd.DataFrame:
    """Fill gross_profit, ebit, ebitda, free_cash_flow and total_debt from components when missing."""
    if df.empty:
        return df
    extra: list[dict[str, Any]] = []
    for (period_end, period_type), g in df.groupby(["period_end", "period_type"]):
        have = dict(zip(g["item"], g["value"], strict=True))
        filed = max(g["filed_at"])
        est = bool(g["filed_at_estimated"].any())

        def add(
            item: str,
            value: float | None,
            *,
            _pe=period_end,
            _pt=period_type,
            _filed=filed,
            _est=est,
            _have=have,
        ) -> None:
            if value is not None and np.isfinite(value) and item not in _have:
                extra.append(
                    {
                        "instrument_id": inst.instrument_id,
                        "statement": ITEMS[item],
                        "item": item,
                        "period_end": _pe,
                        "period_type": _pt,
                        "value": float(value),
                        "currency": inst.currency.value,
                        "filed_at": _filed,
                        "filed_at_estimated": _est,
                        "vendor": vendor,
                    }
                )
                _have[item] = float(value)

        if "revenue" in have and "cost_of_revenue" in have:
            add("gross_profit", have["revenue"] - have["cost_of_revenue"])
        if "operating_income" in have:
            add("ebit", have["operating_income"])
        if "ebit" in have and "depreciation_amortization" in have:
            add("ebitda", have["ebit"] + have["depreciation_amortization"])
        if "operating_cash_flow" in have and "capex" in have:
            add("free_cash_flow", have["operating_cash_flow"] - abs(have["capex"]))
        if "long_term_debt" in have or "short_term_debt" in have:
            add("total_debt", have.get("long_term_debt", 0.0) + have.get("short_term_debt", 0.0))
    if extra:
        df = pd.concat([df, _rows_to_frame(extra)], ignore_index=True)
    return df.sort_values(["statement", "item", "period_end"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- query object


@dataclass
class StatementSet:
    """One instrument's canonical rows with point-in-time accessors."""

    instrument_id: str
    rows: pd.DataFrame

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, instrument_id: str | None = None) -> StatementSet:
        if rows.empty:
            return cls(instrument_id or "", _empty())
        iid = instrument_id or str(rows["instrument_id"].iloc[0])
        return cls(iid, rows[rows["instrument_id"] == iid].reset_index(drop=True))

    def as_of(self, when: date | str | None) -> StatementSet:
        """Rows that had been filed by `when` (all rows when None)."""
        if when is None or self.rows.empty:
            return self
        day = to_date(when)
        return StatementSet(
            self.instrument_id, self.rows[self.rows["filed_at"] <= day].reset_index(drop=True)
        )

    def series(self, item: str, period_type: str = "annual") -> pd.Series:
        sel = self.rows[(self.rows["item"] == item) & (self.rows["period_type"] == period_type)]
        return pd.Series(
            sel["value"].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(pd.to_datetime(sel["period_end"])),
            name=item,
        ).sort_index()

    def latest(self, item: str, period_type: str = "annual") -> float | None:
        s = self.series(item, period_type)
        return None if s.empty else float(s.iloc[-1])

    def quarters(self, item: str) -> int:
        """Number of quarterly observations known for `item`."""
        return int(((self.rows["item"] == item) & (self.rows["period_type"] == "quarterly")).sum())

    def latest_period_end(self, period_type: str = "annual") -> date | None:
        sel = self.rows[self.rows["period_type"] == period_type]
        return None if sel.empty else max(sel["period_end"])

    def ttm(self, item: str) -> float | None:
        """Trailing twelve months: sum of the last four quarters for flow items, latest quarter for stocks;
        falls back to the latest annual figure when fewer than four quarters are known."""
        q = self.series(item, "quarterly")
        if item not in FLOW_ITEMS:
            return float(q.iloc[-1]) if not q.empty else self.latest(item)
        if len(q) >= 4:
            return float(q.iloc[-4:].sum())
        return self.latest(item)

    def yoy_growth(self, item: str, period_type: str = "annual") -> float | None:
        s = self.series(item, period_type)
        if len(s) < 2 or s.iloc[-2] == 0:
            return None
        return float(s.iloc[-1] / abs(s.iloc[-2]) - 1.0)

    def items(self) -> list[str]:
        return sorted(self.rows["item"].unique()) if not self.rows.empty else []

    def filing_lag_estimated(self) -> bool:
        return bool(self.rows["filed_at_estimated"].any()) if not self.rows.empty else False
