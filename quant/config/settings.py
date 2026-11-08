"""Layered settings: package defaults -> user YAML -> .env -> environment -> explicit overrides.

Secrets (API keys) are only ever read from the environment or `.env`, never from YAML, and are stored as
`SecretStr` so they do not leak into logs or `quant config` output.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from quant.core.catalog import INSTRUMENTS_YAML
from quant.core.errors import ConfigError
from quant.core.types import Currency

PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_YAML = PACKAGE_CONFIG_DIR / "default.yaml"
FEES_YAML = PACKAGE_CONFIG_DIR / "fees.yaml"

_yaml_paths: ContextVar[tuple[Path, ...]] = ContextVar("quant_yaml_paths", default=(DEFAULT_YAML,))


class PathsSettings(BaseModel):
    data_dir: Path = Path("data")

    @property
    def parquet_dir(self) -> Path:
        return self.data_dir / "parquet"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "quant.duckdb"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    def ensure(self) -> PathsSettings:
        for p in (self.data_dir, self.parquet_dir, self.reports_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


class TradingSettings(BaseModel):
    live_enabled: bool = Field(default=False, description="Live-trading gate flag. v1 has no live broker.")
    default_broker: Literal["sim", "alpaca"] = "sim"
    base_currency: Currency = Currency.USD
    cash_policy: Literal["convert_at_fill", "hold_foreign"] = "convert_at_fill"
    fx_cost_bps: float = Field(default=2.0, ge=0)


class DataSettings(BaseModel):
    adapter_priority: list[str] = Field(default_factory=lambda: ["yfinance", "stooq"])
    cache_ttl_hours: float = Field(default=12.0, ge=0)
    drop_unfinished_session: bool = True
    default_start: str = "2000-01-01"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Merged YAML files registered for the current `load_settings()` call (defaults first)."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        merged: dict[str, Any] = {}
        for path in _yaml_paths.get():
            merged = _deep_merge(merged, _read_yaml(path))
        unknown = set(merged) - set(settings_cls.model_fields)
        if unknown:
            raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
        self._data = merged

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


class QuantSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    fees_file: Path = FEES_YAML
    instruments_file: Path = INSTRUMENTS_YAML

    # API credentials: environment / .env only (QUANT_FRED_API_KEY=...). Never put these in YAML.
    fred_api_key: SecretStr | None = None
    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    polygon_api_key: SecretStr | None = None
    tiingo_api_key: SecretStr | None = None
    ndl_api_key: SecretStr | None = None
    companies_house_api_key: SecretStr | None = None
    sec_user_agent: str | None = Field(default=None, description="EDGAR requires 'Name contact@email' here.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest priority first: explicit kwargs > environment > .env > YAML (user over defaults).
        return (init_settings, env_settings, dotenv_settings, YamlSettingsSource(settings_cls), file_secret_settings)


def load_settings(config_file: str | Path | None = None, **overrides: Any) -> QuantSettings:
    """Build settings from the package defaults plus an optional user YAML, then .env and environment."""
    paths: tuple[Path, ...] = (DEFAULT_YAML,)
    if config_file is not None:
        paths += (Path(config_file),)
    token = _yaml_paths.set(paths)
    try:
        return QuantSettings(**overrides)
    finally:
        _yaml_paths.reset(token)
