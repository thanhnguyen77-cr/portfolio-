from __future__ import annotations

import pytest

from quant.core.calendars import get_calendar
from quant.core.catalog import InstrumentCatalog, load_default_catalog
from quant.core.errors import ConfigError
from quant.core.types import AssetClass, Currency, Instrument, Market


@pytest.fixture(scope="module")
def catalog() -> InstrumentCatalog:
    return load_default_catalog()


def test_default_catalog_loads_with_unique_ids(catalog):
    assert len(catalog) > 50
    ids = [i.instrument_id for i in catalog]
    assert len(ids) == len(set(ids))


def test_every_instrument_resolves_to_a_calendar(catalog):
    for inst in catalog:
        assert get_calendar(inst.calendar_code).is_session.__call__ is not None


def test_vendor_symbol_lookup_and_pence(catalog):
    vod = catalog.by_vendor_symbol("yahoo", "vod.l")
    assert vod is not None
    assert vod.instrument_id == "XLON_VOD"
    assert vod.quoted_in_pence and vod.currency is Currency.GBP
    assert vod.scale_price(7250.0) == pytest.approx(72.5)
    assert catalog.by_vendor_symbol("yahoo", "DOES-NOT-EXIST") is None
    assert catalog["FX_GBPUSD"].currency is Currency.USD
    assert catalog.by_vendor_symbol("fred", "DGS10").symbol == "US10Y"


def test_pence_quoted_instruments_are_gbp(catalog):
    for inst in catalog.filter(market=Market.XLON):
        if inst.price_scale == 0.01:
            assert inst.currency is Currency.GBP


def test_indices_rates_and_continuous_futures_are_not_tradable(catalog):
    for ac in (AssetClass.INDEX, AssetClass.RATE, AssetClass.COMMODITY):
        for inst in catalog.filter(asset_class=ac):
            assert not inst.tradable, inst.instrument_id


def test_filter_and_by_symbol(catalog):
    fx = catalog.filter(asset_class="fx")
    assert all(i.market is Market.FX for i in fx) and len(fx) >= 10
    assert catalog.by_symbol("SPY")[0].instrument_id == "XNYS_SPY"
    assert catalog.by_symbol("SPY", market="XLON") == []


def test_duplicate_rejected_unless_replace():
    inst = Instrument(symbol="X", asset_class="fx", market="FX", currency="USD")
    cat = InstrumentCatalog([inst])
    with pytest.raises(ConfigError, match="duplicate"):
        cat.add(inst)
    cat.add(inst, replace=True)
    assert len(cat) == 1
    with pytest.raises(KeyError):
        cat["NOPE"]


def test_invalid_yaml_raises_config_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("instruments:\n  - {symbol: A, asset_class: equity, market: LSE, currency: GBP}\n")
    with pytest.raises(ConfigError, match="invalid instrument catalogue"):
        InstrumentCatalog.from_yaml(bad)
    with pytest.raises(ConfigError, match="not found"):
        InstrumentCatalog.from_yaml(tmp_path / "missing.yaml")
