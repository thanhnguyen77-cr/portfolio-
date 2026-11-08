"""Licensed-vendor stubs: the adapter contract, documented wiring steps, and nothing else.

A stub declares which capabilities the vendor WOULD provide (`intended_capabilities`), but `capabilities()`
returns an empty set until the vendor SDK is importable and credentials are present, so the DataService
never routes a request to an unconfigured licensed feed by accident. Every data method raises
`AdapterNotConfigured` with the concrete steps (install, entitlement, identifiers) for that vendor.
Implementing a vendor means subclassing, keeping `name`/`identifier_key`, and overriding the data methods
to return `AdapterResult` frames in the canonical schemas (see quant.data.adapters._bars).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from datetime import date

from quant.core.errors import AdapterNotConfigured
from quant.core.types import Instrument
from quant.data.adapters.base import AdapterResult, BaseAdapter, Capability, RateLimit


class LicensedStub(BaseAdapter):
    name = "licensed"
    identifier_key: str | None = None
    vendor_label = "a licensed vendor"
    sdk_module: str | None = None  # importable module that proves the SDK is installed
    install_hint = "install the vendor SDK"
    auth_hint = "configure credentials / entitlements"
    identifier_hint = "populate Instrument.identifiers with the vendor's symbology"
    docs_url = ""
    intended_capabilities: frozenset[Capability] = frozenset()
    rate_limit = RateLimit(note="governed by the vendor contract")

    def __init__(self, credentials: dict[str, str] | None = None) -> None:
        super().__init__()
        self.credentials = dict(credentials or {})

    # --- configuration state -------------------------------------------------------------------
    def sdk_available(self) -> bool:
        if not self.sdk_module:
            return False
        top = self.sdk_module.split(".")[0]
        try:
            if importlib.util.find_spec(top) is None:
                return False
            return importlib.util.find_spec(self.sdk_module) is not None
        except (ImportError, ValueError):
            return False

    def configured(self) -> bool:
        """True only when the SDK imports AND a subclass has implemented the data methods."""
        return False

    def capabilities(self) -> frozenset[Capability]:
        return self.intended_capabilities if self.configured() else frozenset()

    def wiring_steps(self) -> list[str]:
        steps = [
            f"1. {self.install_hint}"
            + ("" if self.sdk_available() else " (SDK not found in this environment)"),
            f"2. {self.auth_hint}",
            f"3. {self.identifier_hint} (identifiers key: {self.identifier_key!r})",
            f"4. subclass {type(self).__name__} and implement the data methods for the capabilities "
            f"{sorted(c.value for c in self.intended_capabilities)}",
        ]
        if self.docs_url:
            steps.append(f"docs: {self.docs_url}")
        return steps

    def _not_configured(self, what: str) -> AdapterNotConfigured:
        return AdapterNotConfigured(
            f"{self.vendor_label} ({self.name}) is not wired for {what}. " + " ".join(self.wiring_steps())
        )

    # --- data methods: raise with guidance -------------------------------------------------------
    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        raise self._not_configured("bars")

    def get_corporate_actions(
        self, instruments: Sequence[Instrument], start: date, end: date
    ) -> AdapterResult:
        raise self._not_configured("corporate actions")

    def get_macro_series(self, series_ids: Sequence[str], start: date, end: date) -> AdapterResult:
        raise self._not_configured("macro series")

    def get_fundamentals(self, instruments: Sequence[Instrument]) -> AdapterResult:
        raise self._not_configured("fundamentals")

    def get_earnings(self, instruments: Sequence[Instrument]) -> AdapterResult:
        raise self._not_configured("earnings")


class BloombergAdapter(LicensedStub):
    """Bloomberg Desktop/Server API via `blpapi` (//blp/refdata Historical/ReferenceDataRequest)."""

    name = "bloomberg"
    identifier_key = "bbg"
    vendor_label = "Bloomberg"
    sdk_module = "blpapi"
    install_hint = (
        "pip install blpapi --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/"
    )
    auth_hint = (
        "run next to a logged-in Terminal (Desktop API, localhost:8194) or a SAPI/B-PIPE with entitlements"
    )
    identifier_hint = "store Bloomberg tickers such as 'VOD LN Equity' or 'AAPL US Equity'"
    docs_url = "https://www.bloomberg.com/professional/support/api-library/"
    intended_capabilities = frozenset(
        {
            Capability.BARS_DAILY,
            Capability.CORPORATE_ACTIONS,
            Capability.FUNDAMENTALS,
            Capability.EARNINGS,
            Capability.MACRO,
            Capability.FX,
        }
    )
    field_hints = {
        "open": "PX_OPEN",
        "high": "PX_HIGH",
        "low": "PX_LOW",
        "close": "PX_LAST",
        "volume": "PX_VOLUME",
    }


class RefinitivAdapter(LicensedStub):
    """LSEG Refinitiv Data Library (`refinitiv-data`): rd.get_history / rd.get_data with RICs."""

    name = "refinitiv"
    identifier_key = "ric"
    vendor_label = "LSEG Refinitiv"
    sdk_module = "refinitiv.data"
    install_hint = "pip install refinitiv-data"
    auth_hint = "Workspace desktop session or RDP app key + credentials in refinitiv-data.config.json"
    identifier_hint = "store RICs such as 'VOD.L' or 'AAPL.O'"
    docs_url = (
        "https://developers.lseg.com/en/api-catalog/refinitiv-data-platform/refinitiv-data-library-for-python"
    )
    intended_capabilities = frozenset(
        {
            Capability.BARS_DAILY,
            Capability.CORPORATE_ACTIONS,
            Capability.FUNDAMENTALS,
            Capability.EARNINGS,
            Capability.FX,
        }
    )


class FactSetAdapter(LicensedStub):
    """FactSet Content APIs (`fds.sdk.*`: FactSetPrices, FactSetFundamentals) keyed by FactSet ids."""

    name = "factset"
    identifier_key = "factset"
    vendor_label = "FactSet"
    sdk_module = "fds.sdk.FactSetPrices"
    install_hint = "pip install fds.sdk.FactSetPrices fds.sdk.FactSetFundamentals"
    auth_hint = "FactSet API username-serial and API key with Prices/Fundamentals entitlements"
    identifier_hint = "store FactSet ids or tickers such as 'VOD-GB' / 'AAPL-US'"
    docs_url = "https://developer.factset.com/"
    intended_capabilities = frozenset(
        {Capability.BARS_DAILY, Capability.CORPORATE_ACTIONS, Capability.FUNDAMENTALS}
    )


class CapIQAdapter(LicensedStub):
    """S&P Capital IQ: no public Python SDK; data arrives through the Capital IQ Pro API or Excel plug-in."""

    name = "capiq"
    identifier_key = "capiq"
    vendor_label = "S&P Capital IQ"
    sdk_module = None
    install_hint = (
        "no pip package: use the Capital IQ Pro API (REST, contract required) or exported Excel sheets"
    )
    auth_hint = "API credentials issued by S&P Global under your licence"
    identifier_hint = "store Capital IQ company ids such as 'IQ24937' or exchange:ticker pairs"
    docs_url = "https://www.spglobal.com/marketintelligence/en/solutions/capital-iq-pro-api"
    intended_capabilities = frozenset({Capability.FUNDAMENTALS, Capability.EARNINGS})


LICENSED_STUBS: tuple[type[LicensedStub], ...] = (
    BloombergAdapter,
    RefinitivAdapter,
    FactSetAdapter,
    CapIQAdapter,
)
