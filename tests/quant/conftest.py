"""Shared fixtures for the quant test-suite.

Tests never touch the network. The autouse `_no_network` fixture blocks IPv4/IPv6 sockets, points the HTTP
proxy variables at a dead port, blocks `requests` at the adapter layer and stubs yfinance's entry points
(yfinance uses libcurl via curl_cffi, which bypasses Python sockets). Mark a test `@pytest.mark.live` to opt
out; those are deselected by default (see `addopts` in pyproject.toml).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

_RealSocket = socket.socket


class _GuardedSocket(_RealSocket):
    def _refuse(self, verb: str, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError(
                f"network access is disabled in tests (attempted {verb} to {address!r}); "
                "patch the adapter seam, or mark the test @pytest.mark.live"
            )

    def connect(self, address):  # type: ignore[override]
        self._refuse("connect", address)
        return super().connect(address)

    def connect_ex(self, address):  # type: ignore[override]
        self._refuse("connect_ex", address)
        return super().connect_ex(address)


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if request.node.get_closest_marker("live") is not None:
        yield
        return
    monkeypatch.setattr(socket, "socket", _GuardedSocket)
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")
    try:
        import yfinance
    except ImportError:
        pass
    else:

        def _blocked_yf(*args, **kwargs):
            raise RuntimeError("yfinance network call blocked in tests; patch the adapter seam instead")

        for name in ("Ticker", "Tickers", "download"):
            if hasattr(yfinance, name):
                monkeypatch.setattr(yfinance, name, _blocked_yf)
    try:
        import requests.adapters
    except ImportError:
        pass
    else:

        def _blocked_send(self, req, *args, **kwargs):
            raise RuntimeError(f"HTTP blocked in tests: {req.method} {req.url}")

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _blocked_send)
    yield


@pytest.fixture
def tmp_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Settings rooted in a temp directory, with cwd moved there so no stray .env is read."""
    monkeypatch.chdir(tmp_path)
    from quant.config.settings import load_settings

    settings = load_settings(paths={"data_dir": str(tmp_path / "data")})
    settings.paths.ensure()
    return settings
