"""Instrument catalogue: the seed list in `quant/config/instruments.yaml` plus runtime registrations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from quant.core.errors import ConfigError
from quant.core.types import AssetClass, Instrument, Market

INSTRUMENTS_YAML = Path(__file__).resolve().parents[1] / "config" / "instruments.yaml"


class _CatalogFile(BaseModel):
    instruments: list[Instrument]


class InstrumentCatalog:
    """Dict-like registry keyed by `instrument_id`, with vendor-symbol reverse lookup."""

    def __init__(self, instruments: Iterable[Instrument] = ()) -> None:
        self._by_id: dict[str, Instrument] = {}
        self._by_vendor: dict[tuple[str, str], str] = {}
        for inst in instruments:
            self.add(inst)

    def add(self, inst: Instrument, *, replace: bool = False) -> None:
        if inst.instrument_id in self._by_id and not replace:
            raise ConfigError(f"duplicate instrument_id {inst.instrument_id!r}")
        self._by_id[inst.instrument_id] = inst
        for vendor, sym in inst.identifiers.items():
            self._by_vendor[(vendor.lower(), sym.upper())] = inst.instrument_id

    def __getitem__(self, instrument_id: str) -> Instrument:
        try:
            return self._by_id[instrument_id]
        except KeyError:
            raise KeyError(f"unknown instrument_id {instrument_id!r}") from None

    def get(self, instrument_id: str, default: Instrument | None = None) -> Instrument | None:
        return self._by_id.get(instrument_id, default)

    def __contains__(self, instrument_id: object) -> bool:
        return instrument_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._by_id.values())

    def by_vendor_symbol(self, vendor: str, symbol: str) -> Instrument | None:
        iid = self._by_vendor.get((vendor.lower(), symbol.upper()))
        return self._by_id[iid] if iid else None

    def by_symbol(self, symbol: str, market: Market | str | None = None) -> list[Instrument]:
        sym = symbol.upper()
        out = [i for i in self._by_id.values() if i.symbol.upper() == sym]
        if market is not None:
            out = [i for i in out if i.market == Market(market)]
        return out

    def filter(
        self,
        *,
        asset_class: AssetClass | str | None = None,
        market: Market | str | None = None,
        tradable: bool | None = None,
    ) -> list[Instrument]:
        out = list(self._by_id.values())
        if asset_class is not None:
            out = [i for i in out if i.asset_class == AssetClass(asset_class)]
        if market is not None:
            out = [i for i in out if i.market == Market(market)]
        if tradable is not None:
            out = [i for i in out if i.tradable is tradable]
        return out

    @classmethod
    def from_yaml(cls, path: Path | str) -> InstrumentCatalog:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"instrument catalogue not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        try:
            parsed = _CatalogFile.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"invalid instrument catalogue {path}:\n{exc}") from exc
        return cls(parsed.instruments)


def load_default_catalog() -> InstrumentCatalog:
    return InstrumentCatalog.from_yaml(INSTRUMENTS_YAML)
