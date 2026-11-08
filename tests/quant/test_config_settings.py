from __future__ import annotations

import pytest
from pydantic import ValidationError

from quant.config.settings import DEFAULT_YAML, load_settings
from quant.core.errors import ConfigError
from quant.core.types import Currency


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in list(__import__("os").environ):
        if var.startswith("QUANT_"):
            monkeypatch.delenv(var)


def test_defaults():
    s = load_settings()
    assert DEFAULT_YAML.exists()
    assert s.trading.live_enabled is False
    assert s.trading.default_broker == "sim"
    assert s.trading.base_currency is Currency.USD
    assert s.trading.cash_policy == "convert_at_fill"
    assert s.data.adapter_priority == ["yfinance", "stooq", "fred", "boe", "sec_edgar"]
    assert s.fees_file.exists() and s.instruments_file.exists()
    assert s.fred_api_key is None


def test_environment_overrides_nested_fields(monkeypatch):
    monkeypatch.setenv("QUANT_TRADING__BASE_CURRENCY", "GBP")
    monkeypatch.setenv("QUANT_DATA__CACHE_TTL_HOURS", "1.5")
    s = load_settings()
    assert s.trading.base_currency is Currency.GBP
    assert s.data.cache_ttl_hours == 1.5


def test_yaml_layer_and_env_precedence(tmp_path, monkeypatch):
    cfg = tmp_path / "my.yaml"
    cfg.write_text(
        "trading:\n  base_currency: GBP\n  fx_cost_bps: 5\ndata:\n  adapter_priority: [stooq]\n",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.trading.base_currency is Currency.GBP
    assert s.trading.fx_cost_bps == 5.0
    assert s.data.adapter_priority == ["stooq"]
    assert s.data.cache_ttl_hours == 12.0  # untouched default survives the merge

    monkeypatch.setenv("QUANT_TRADING__FX_COST_BPS", "7")
    s2 = load_settings(cfg)
    assert s2.trading.fx_cost_bps == 7.0  # env beats YAML ...
    assert s2.trading.base_currency is Currency.GBP  # ... without clobbering sibling YAML keys


def test_explicit_overrides_beat_environment(monkeypatch):
    monkeypatch.setenv("QUANT_TRADING__BASE_CURRENCY", "GBP")
    s = load_settings(trading={"base_currency": "EUR"})
    assert s.trading.base_currency is Currency.EUR


def test_unknown_yaml_key_rejected(tmp_path):
    cfg = tmp_path / "typo.yaml"
    cfg.write_text("tradng:\n  live_enabled: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level config keys"):
        load_settings(cfg)


def test_missing_yaml_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_invalid_values_rejected(monkeypatch):
    monkeypatch.setenv("QUANT_TRADING__DEFAULT_BROKER", "ibkr")
    with pytest.raises(ValidationError):
        load_settings()


def test_secrets_are_masked(monkeypatch):
    monkeypatch.setenv("QUANT_FRED_API_KEY", "topsecret-123")
    s = load_settings()
    assert s.fred_api_key is not None
    assert s.fred_api_key.get_secret_value() == "topsecret-123"
    assert "topsecret" not in repr(s)
    assert "topsecret" not in s.model_dump_json()


def test_paths_derive_and_ensure(tmp_path):
    s = load_settings(paths={"data_dir": str(tmp_path / "d")})
    s.paths.ensure()
    assert (tmp_path / "d" / "parquet").is_dir()
    assert (tmp_path / "d" / "reports").is_dir()
    assert s.paths.duckdb_path == tmp_path / "d" / "quant.duckdb"
