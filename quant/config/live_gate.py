"""The live-trading gate. Every live path in the package must go through `require_live_confirmation`.

Three independent conditions must all hold: the config flag `trading.live_enabled`, the environment
variable `QUANT_I_UNDERSTAND_LIVE_TRADING=1`, and an interactive typed confirmation. Even then, this version
raises `NotImplementedError`, because no live broker exists. Paper trading never calls this function.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import NoReturn

from quant.config.settings import QuantSettings
from quant.core.errors import LiveTradingDisabled

LIVE_ENV_VAR = "QUANT_I_UNDERSTAND_LIVE_TRADING"
CONFIRMATION_PHRASE = "I understand live trading risks real money"


def require_live_confirmation(settings: QuantSettings, confirm: Callable[[str], str] | None = None) -> NoReturn:
    """Raise `LiveTradingDisabled` unless every gate passes; then raise `NotImplementedError` (v1)."""
    if not settings.trading.live_enabled:
        raise LiveTradingDisabled(
            "live trading is disabled: trading.live_enabled is false (the default). Paper trading only."
        )
    if os.environ.get(LIVE_ENV_VAR) != "1":
        raise LiveTradingDisabled(f"live trading requires the environment variable {LIVE_ENV_VAR}=1")
    if confirm is None:
        raise LiveTradingDisabled("live trading requires an interactive confirmation and none is available")
    typed = confirm(f"Type exactly '{CONFIRMATION_PHRASE}' to continue: ")
    if typed.strip() != CONFIRMATION_PHRASE:
        raise LiveTradingDisabled("confirmation phrase did not match; live trading aborted")
    raise NotImplementedError(
        "All live-trading gates passed, but this version has no live broker implementation. Paper trading only."
    )
