"""Transaction cost model shared by the backtester and the paper SimBroker.

Explicit, itemised costs per fill (commission, half-spread, slippage, transaction taxes/levies, FX
conversion) plus daily borrow on short positions. Fill prices themselves stay inside the bar; costs are
charged separately so every report can show where the money went.

Transaction taxes come from the date-effective schedule in `quant/config/fees.yaml`:
  UK SDRT 0.5 % on purchases of UK-incorporated ordinaries (classes in the rule's applies_to; UNKNOWN is
  taxed conservatively), Irish stamp duty on `irish`, the PTM levy on UK equity trades above the threshold
  (both sides), and US SEC Section 31 + FINRA TAF on sells of US equities/ETFs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quant.config.fees import FeeSchedule, load_fee_schedule, rule_in_force
from quant.core.types import AssetClass, Instrument, Market, Side

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostBreakdown:
    currency: str
    commission: float = 0.0
    spread: float = 0.0
    slippage: float = 0.0
    taxes: dict[str, float] | None = None
    fx_cost: float = 0.0

    @property
    def tax_total(self) -> float:
        return float(sum((self.taxes or {}).values()))

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage + self.tax_total + self.fx_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "commission": self.commission,
            "spread": self.spread,
            "slippage": self.slippage,
            "taxes": dict(self.taxes or {}),
            "fx_cost": self.fx_cost,
            "total": self.total,
        }


class CostModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    commission_per_share: float = Field(default=0.0, ge=0)
    commission_per_order: float = Field(default=0.0, ge=0)
    commission_bps: float = Field(default=0.0, ge=0)
    half_spread_bps: float = Field(
        default=2.0, ge=0, description="Default half bid-ask spread paid per fill."
    )
    half_spread_bps_by_instrument: dict[str, float] = Field(default_factory=dict)
    slippage_bps: float = Field(default=2.0, ge=0)
    impact_coeff: float = Field(
        default=0.0, ge=0, description="Extra slippage multiplier per unit participation."
    )
    fx_cost_bps: float = Field(default=2.0, ge=0)
    borrow_bps_annual: float = Field(default=50.0, ge=0, description="Annual stock-borrow cost on shorts.")
    apply_transaction_taxes: bool = True
    fee_schedule: FeeSchedule | None = None

    def schedule(self) -> FeeSchedule:
        if self.fee_schedule is None:
            object.__setattr__(self, "fee_schedule", load_fee_schedule())
        return self.fee_schedule  # type: ignore[return-value]

    # ------------------------------------------------------------------ per fill
    def fill_costs(
        self,
        inst: Instrument,
        side: Side | str,
        qty: float,
        price: float,
        on: date,
        participation: float | None = None,
    ) -> CostBreakdown:
        side = Side(side)
        consideration = abs(qty) * price
        commission = self.commission_per_order + self.commission_per_share * abs(qty)
        commission += consideration * self.commission_bps / 1e4
        half_spread_bps = self.half_spread_bps_by_instrument.get(inst.instrument_id, self.half_spread_bps)
        spread = consideration * half_spread_bps / 1e4
        impact = 1.0 + self.impact_coeff * (participation or 0.0)
        slippage = consideration * self.slippage_bps / 1e4 * impact
        taxes = self.transaction_taxes(inst, side, abs(qty), consideration, on)
        return CostBreakdown(
            currency=inst.currency.value, commission=commission, spread=spread, slippage=slippage, taxes=taxes
        )

    def transaction_taxes(
        self, inst: Instrument, side: Side | str, qty: float, consideration: float, on: date
    ) -> dict[str, float]:
        side = Side(side)
        out: dict[str, float] = {}
        if not self.apply_transaction_taxes:
            return out
        sched = self.schedule()
        if inst.market is Market.XLON:
            if side is Side.BUY:
                sdrt = rule_in_force(sched.uk.sdrt, on)
                if sdrt and inst.stamp_duty_class in sdrt.applies_to and side in sdrt.sides:
                    out["sdrt"] = consideration * sdrt.rate
                irish = rule_in_force(sched.uk.irish_stamp_duty, on)
                if irish and inst.stamp_duty_class in irish.applies_to and side in irish.sides:
                    out["irish_stamp_duty"] = consideration * irish.rate
            if inst.asset_class is AssetClass.EQUITY:
                ptm = rule_in_force(sched.uk.ptm_levy, on)
                if ptm and side in ptm.sides and consideration > ptm.threshold:
                    out["ptm_levy"] = ptm.amount
        elif inst.market in (Market.XNYS, Market.XNAS) and inst.asset_class in (
            AssetClass.EQUITY,
            AssetClass.ETF,
        ):
            if side is Side.SELL:
                s31 = rule_in_force(sched.us.sec_section31, on)
                if s31 and side in s31.sides:
                    out["sec_section31"] = consideration * s31.rate_per_unit
                taf = rule_in_force(sched.us.finra_taf, on)
                if taf and side in taf.sides:
                    fee = qty * taf.per_share
                    if taf.max_per_trade is not None:
                        fee = min(fee, taf.max_per_trade)
                    out["finra_taf"] = fee
        return {k: v for k, v in out.items() if v > 0}

    # ------------------------------------------------------------------ daily / conversion
    def daily_borrow_cost(self, short_notional_abs: float) -> float:
        return abs(short_notional_abs) * self.borrow_bps_annual / 1e4 / TRADING_DAYS

    def fx_conversion_cost(self, amount: float) -> float:
        return abs(amount) * self.fx_cost_bps / 1e4

    def describe(self) -> dict[str, Any]:
        d = self.model_dump(exclude={"fee_schedule"})
        d["unverified_fee_rules"] = self.schedule().unverified() if self.apply_transaction_taxes else []
        return d
