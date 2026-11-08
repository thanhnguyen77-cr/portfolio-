from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant.core.types import (
    BAR_COLUMNS,
    AssetClass,
    Currency,
    Instrument,
    Market,
    StampDutyClass,
    bar_schema,
    make_instrument_id,
    slugify_symbol,
)


@pytest.mark.parametrize(
    ("raw", "slug"),
    [("BRK-B", "BRK_B"), ("brk.b", "BRK_B"), ("^GSPC", "GSPC"), ("GBPUSD=X", "GBPUSD_X"), (" vod ", "VOD")],
)
def test_slugify(raw, slug):
    assert slugify_symbol(raw) == slug


def test_slugify_empty_raises():
    with pytest.raises(ValueError):
        slugify_symbol("^^")


def test_make_instrument_id():
    assert make_instrument_id("XLON", "VOD") == "XLON_VOD"
    assert make_instrument_id(Market.XNYS, "BRK-B") == "XNYS_BRK_B"
    with pytest.raises(ValueError):
        make_instrument_id("NOPE", "X")


def _vod() -> Instrument:
    return Instrument(
        symbol="VOD",
        asset_class="equity",
        market="XLON",
        currency="GBP",
        price_scale=0.01,
        identifiers={"yahoo": "VOD.L"},
    )


def test_instrument_derives_id_and_handles_pence():
    inst = _vod()
    assert inst.instrument_id == "XLON_VOD"
    assert inst.quoted_in_pence
    assert inst.calendar_code == "XLON"
    assert inst.vendor_symbol("yahoo") == "VOD.L"
    assert inst.vendor_symbol("bbg") == "VOD"
    assert inst.scale_price(7250.0) == pytest.approx(72.5)
    assert inst.stamp_duty_class is StampDutyClass.UNKNOWN
    assert inst.currency is Currency.GBP


def test_instrument_explicit_id_is_kept():
    inst = Instrument(instrument_id="CUSTOM", symbol="X", asset_class="fx", market="FX", currency="USD")
    assert inst.instrument_id == "CUSTOM"


def test_instrument_is_frozen():
    inst = _vod()
    with pytest.raises(ValidationError):
        inst.symbol = "ABC"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"price_scale": 0},
        {"lot_size": 0},
        {"currency": "GBX"},
        {"market": "LSE"},
        {"unexpected": 1},
    ],
)
def test_instrument_validation(bad):
    base = dict(symbol="VOD", asset_class="equity", market="XLON", currency="GBP")
    base.update(bad)
    with pytest.raises(ValidationError):
        Instrument(**base)


def test_bar_schema_matches_columns():
    schema = bar_schema()
    assert tuple(schema.names) == BAR_COLUMNS
    assert str(schema.field("ts_close_utc").type) == "timestamp[ns, tz=UTC]"
    assert str(schema.field("session_date").type) == "date32[day]"


def test_enums_are_plain_strings():
    assert AssetClass.EQUITY == "equity"
    assert Currency("GBP") is Currency.GBP
    assert StampDutyClass.UK_ORDINARY.value == "uk_ordinary"
    assert Market.XLON.calendar_code == "XLON"
