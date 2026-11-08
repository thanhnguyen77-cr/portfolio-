from __future__ import annotations

from datetime import date

import pytest

from quant.config.fees import FeeSchedule, load_fee_schedule, rule_in_force
from quant.core.errors import ConfigError
from quant.core.types import Side, StampDutyClass


@pytest.fixture(scope="module")
def schedule() -> FeeSchedule:
    return load_fee_schedule()


def test_schedule_has_all_tables(schedule):
    assert schedule.uk.sdrt and schedule.uk.irish_stamp_duty and schedule.uk.ptm_levy
    assert schedule.us.sec_section31 and schedule.us.finra_taf


def test_sdrt_rule(schedule):
    rule = rule_in_force(schedule.uk.sdrt, date(2024, 1, 1))
    assert rule is not None
    assert rule.rate == 0.005
    assert StampDutyClass.UK_ORDINARY in rule.applies_to
    assert StampDutyClass.UNKNOWN in rule.applies_to  # conservative default
    assert StampDutyClass.EXEMPT not in rule.applies_to
    assert rule.sides == [Side.BUY]


def test_ptm_levy_is_date_effective_and_two_sided(schedule):
    old = rule_in_force(schedule.uk.ptm_levy, date(2020, 1, 1))
    new = rule_in_force(schedule.uk.ptm_levy, date(2024, 1, 1))
    assert old is not None and new is not None
    assert (old.amount, new.amount) == (1.0, 1.5)
    assert old.threshold == new.threshold == 10000
    assert Side.SELL in new.sides and Side.BUY in new.sides
    assert new.currency == "GBP"


def test_rule_before_first_effective_date_is_none(schedule):
    assert rule_in_force(schedule.uk.sdrt, date(1980, 1, 1)) is None


def test_unverified_rules_are_reported(schedule):
    unverified = schedule.unverified()
    assert any(name.startswith("uk.ptm_levy@2023") for name in unverified)
    assert any(name.startswith("us.sec_section31") for name in unverified)
    assert not any(name.startswith("uk.sdrt") for name in unverified)


def test_invalid_schedule_rejected(tmp_path):
    bad = tmp_path / "fees.yaml"
    bad.write_text(
        "uk:\n  sdrt:\n    - {effective_from: 2000-01-01, rate: -0.1, applies_to: [uk_ordinary]}\n"
    )
    with pytest.raises(ConfigError, match="invalid fee schedule"):
        load_fee_schedule(bad)
    with pytest.raises(ConfigError, match="not found"):
        load_fee_schedule(tmp_path / "missing.yaml")
