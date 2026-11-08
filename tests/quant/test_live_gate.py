"""Every path through the live-trading gate must raise. There is no live broker in this version."""

from __future__ import annotations

import pytest

from quant.config.live_gate import CONFIRMATION_PHRASE, LIVE_ENV_VAR, require_live_confirmation
from quant.config.settings import load_settings
from quant.core.errors import LiveTradingDisabled


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(LIVE_ENV_VAR, raising=False)
    monkeypatch.delenv("QUANT_TRADING__LIVE_ENABLED", raising=False)


def _always_confirm(prompt: str) -> str:
    return CONFIRMATION_PHRASE


def test_disabled_by_default_even_with_confirmation():
    settings = load_settings()
    with pytest.raises(LiveTradingDisabled, match="live_enabled is false"):
        require_live_confirmation(settings, confirm=_always_confirm)


def test_flag_alone_is_not_enough():
    settings = load_settings(trading={"live_enabled": True})
    with pytest.raises(LiveTradingDisabled, match=LIVE_ENV_VAR):
        require_live_confirmation(settings, confirm=_always_confirm)


def test_flag_and_env_without_interactive_confirmation(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, "1")
    settings = load_settings(trading={"live_enabled": True})
    with pytest.raises(LiveTradingDisabled, match="interactive confirmation"):
        require_live_confirmation(settings)


def test_wrong_phrase_aborts(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, "1")
    settings = load_settings(trading={"live_enabled": True})
    with pytest.raises(LiveTradingDisabled, match="did not match"):
        require_live_confirmation(settings, confirm=lambda _: "yes")


def test_all_gates_passed_still_no_live_broker(monkeypatch):
    monkeypatch.setenv(LIVE_ENV_VAR, "1")
    settings = load_settings(trading={"live_enabled": True})
    with pytest.raises(NotImplementedError, match="no live broker"):
        require_live_confirmation(settings, confirm=_always_confirm)
