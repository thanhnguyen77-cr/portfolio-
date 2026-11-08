"""Date-effective transaction taxes and levies (quant/config/fees.yaml).

Each rule applies from `effective_from` until a later rule supersedes it. Rules carry `verified` so that
values not yet checked against their primary source are reported in backtest Caveats.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from quant.core.errors import ConfigError
from quant.core.types import Side, StampDutyClass

FEES_YAML = Path(__file__).resolve().parent / "fees.yaml"


class DatedRule(BaseModel):
    effective_from: date
    source: str | None = None
    verified: bool = True
    note: str | None = None


class AdValoremTax(DatedRule):
    """Percentage of consideration, e.g. UK SDRT 0.5 % on purchases."""

    rate: float = Field(ge=0)
    applies_to: list[StampDutyClass]
    sides: list[Side] = Field(default_factory=lambda: [Side.BUY])


class FixedLevy(DatedRule):
    """Flat amount above a consideration threshold, e.g. the UK PTM levy."""

    amount: float = Field(ge=0)
    currency: str
    threshold: float = Field(ge=0)
    sides: list[Side]


class PerUnitFee(DatedRule):
    """Fee per unit of consideration, e.g. SEC Section 31 (per dollar of sale proceeds)."""

    rate_per_unit: float = Field(ge=0)
    currency: str
    sides: list[Side]


class PerShareFee(DatedRule):
    """Fee per share with an optional per-trade cap, e.g. FINRA TAF."""

    per_share: float = Field(ge=0)
    max_per_trade: float | None = None
    currency: str
    sides: list[Side]


class UKFees(BaseModel):
    sdrt: list[AdValoremTax] = Field(default_factory=list)
    irish_stamp_duty: list[AdValoremTax] = Field(default_factory=list)
    ptm_levy: list[FixedLevy] = Field(default_factory=list)


class USFees(BaseModel):
    sec_section31: list[PerUnitFee] = Field(default_factory=list)
    finra_taf: list[PerShareFee] = Field(default_factory=list)


class FeeSchedule(BaseModel):
    uk: UKFees = Field(default_factory=UKFees)
    us: USFees = Field(default_factory=USFees)

    def unverified(self) -> list[str]:
        """Names of rules whose values have not been verified against a primary source."""
        out: list[str] = []
        for region_name in ("uk", "us"):
            region = getattr(self, region_name)
            for rule_name, rules in region.model_dump().items():
                for rule in rules:
                    if not rule.get("verified", True):
                        out.append(f"{region_name}.{rule_name}@{rule['effective_from']}")
        return out


R = TypeVar("R", bound=DatedRule)


def rule_in_force(rules: list[R], on: date) -> R | None:
    """The latest rule with `effective_from <= on`, or None when no rule applies yet."""
    applicable = [r for r in rules if r.effective_from <= on]
    return max(applicable, key=lambda r: r.effective_from) if applicable else None


def load_fee_schedule(path: Path | str | None = None) -> FeeSchedule:
    p = Path(path) if path is not None else FEES_YAML
    if not p.exists():
        raise ConfigError(f"fee schedule not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return FeeSchedule.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid fee schedule {p}:\n{exc}") from exc
