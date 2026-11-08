"""The test-suite must be provably offline."""

from __future__ import annotations

import socket

import pytest


def test_inet_connect_is_blocked():
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.2)


def test_unix_sockets_still_work():
    a, b = socket.socketpair()
    try:
        a.sendall(b"ping")
        assert b.recv(4) == b"ping"
    finally:
        a.close()
        b.close()


def test_requests_is_blocked():
    requests = pytest.importorskip("requests")
    with pytest.raises(RuntimeError, match="HTTP blocked"):
        requests.get("http://example.invalid/", timeout=0.2)


def test_yfinance_is_blocked():
    yf = pytest.importorskip("yfinance")
    with pytest.raises(RuntimeError, match="yfinance network call blocked"):
        yf.Ticker("AAPL")


@pytest.mark.live
def test_live_marker_is_opt_in():
    """Only collected with `pytest -m live`; documents the opt-in path."""
    assert True
