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


# --------------------------------------------------------------------------- S1: data-layer helpers


def load_yf_fixture(name: str):
    """A recorded yfinance `Ticker.history()` frame (tz-aware exchange-local index, Yahoo column names)."""
    import pandas as pd

    df = pd.read_csv(FIXTURES / f"yfinance_{name}_history.csv", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    tz = "America/New_York" if name.endswith("AAPL") else "Europe/London"
    df.index = df.index.tz_convert(tz)
    df.index.name = "Date"
    return df


def fake_history(frames: dict, fail: set | None = None, calls: list | None = None):
    """Build a `history_fn` seam replacement serving `frames[symbol]` sliced to [start, end]."""
    import pandas as pd

    fail = fail or set()

    def _history(symbol: str, start, end, interval: str):
        if calls is not None:
            calls.append((symbol, start, end, interval))
        if symbol in fail:
            raise RuntimeError(f"simulated vendor failure for {symbol}")
        if symbol not in frames:
            raise KeyError(f"no fixture for {symbol}")
        df = frames[symbol]
        local = df.index.tz_localize(None).normalize()
        mask = (local >= pd.Timestamp(start)) & (local <= pd.Timestamp(end))
        return df.loc[mask]

    return _history


def make_bars(
    inst,
    start,
    end,
    *,
    seed: int = 0,
    start_price: float = 100.0,
    vol: float = 0.01,
    drift: float = 0.0003,
    vendor: str = "synthetic",
    adjustment_state: str = "raw",
):
    """Synthetic GBM daily bars for `inst` on its real exchange calendar, in canonical form."""
    import numpy as np
    import pandas as pd

    from quant.core.calendars import get_calendar
    from quant.core.time import now
    from quant.data.store import coerce_bars

    cal = get_calendar(inst.calendar_code)
    sessions = cal.sessions(start, end)
    rng = np.random.default_rng(seed)
    n = len(sessions)
    rets = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[start_price, close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = np.exp(rng.normal(np.log(1e6), 0.3, n)).round()
    df = pd.DataFrame(
        {
            "instrument_id": inst.instrument_id,
            "session_date": sessions,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "currency": inst.currency.value,
            "vendor": vendor,
            "adjustment_state": adjustment_state,
            "ts_close_utc": cal.session_closes(start, end).to_numpy(),
            "ingested_at": pd.Timestamp(now()),
        }
    )
    return coerce_bars(df)


class FakeBarsAdapter:
    """In-memory adapter serving pre-generated series; records every call for assertions."""

    name = "fake"

    def __init__(self, series: dict):
        from quant.data.adapters.base import FieldMap, RateLimit

        self.series = series
        self.calls: list[tuple[list[str], object, object]] = []
        self.rate_limit = RateLimit()
        self.field_map = FieldMap()

    def capabilities(self):
        from quant.data.adapters.base import Capability

        return frozenset({Capability.BARS_DAILY})

    def get_bars(self, instruments, start, end, freq: str = "1d"):
        import pandas as pd

        from quant.core.time import now
        from quant.data.adapters.base import AdapterResult, empty_bars

        self.calls.append(([i.instrument_id for i in instruments], start, end))
        frames, errors = [], {}
        for inst in instruments:
            full = self.series.get(inst.instrument_id)
            if full is None:
                errors[inst.instrument_id] = "no fixture"
                continue
            sd = full["session_date"]
            chunk = full[(sd >= pd.Timestamp(start)) & (sd <= pd.Timestamp(end))]
            if chunk.empty:
                errors[inst.instrument_id] = "no data in range"
                continue
            chunk = chunk.copy()
            chunk["vendor"] = self.name
            chunk["ingested_at"] = pd.Timestamp(now())
            frames.append(chunk)
        frame = pd.concat(frames, ignore_index=True) if frames else empty_bars()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=[i.instrument_id for i in instruments],
            params={"start": str(start), "end": str(end), "freq": freq},
        )

    def get_corporate_actions(self, instruments, start, end):
        raise NotImplementedError

    def get_macro_series(self, series_ids, start, end):
        raise NotImplementedError

    def get_fundamentals(self, instruments):
        raise NotImplementedError

    def get_earnings(self, instruments):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def catalog():
    from quant.core.catalog import load_default_catalog

    return load_default_catalog()


# --------------------------------------------------------------------------- S2: anomalies and macro


def plant(bars, *, drop_dates=(), stale=None, outlier=None, denomination=None, jump=None, ohlc=None):
    """Copy of a single-instrument bars frame with planted anomalies.

    drop_dates: sessions to remove (gaps) · stale=(i, n): n equal closes from row i ·
    outlier=(i, mult): one-day spike at row i · denomination=i: x100 from row i on (GBX<->GBP) ·
    jump=(i, ratio): persistent ratio from row i on (split-like) · ohlc=i: high below low at row i.
    """
    px = ["open", "high", "low", "close"]
    out = bars.copy().reset_index(drop=True)
    if drop_dates:
        out = out[~out["session_date"].dt.date.isin(set(drop_dates))].reset_index(drop=True)
    if stale is not None:
        i, n = stale
        out.loc[i : i + n - 1, px] = float(out.loc[i, "close"])
    if outlier is not None:
        i, mult = outlier
        out.loc[i, px] = out.loc[i, px].astype(float) * mult
    if denomination is not None:
        out.loc[denomination:, px] = out.loc[denomination:, px].astype(float) * 100.0
    if jump is not None:
        i, ratio = jump
        out.loc[i:, px] = out.loc[i:, px].astype(float) * ratio
    if ohlc is not None:
        out.loc[ohlc, "high"] = float(out.loc[ohlc, "low"]) * 0.9
    return out


def actions_frame(instrument_id: str, rows, vendor: str = "synthetic"):
    """rows: iterable of (ex_date, action_type, value, currency)."""
    import pandas as pd

    from quant.core.time import now

    df = pd.DataFrame(rows, columns=["ex_date", "action_type", "value", "currency"])
    df.insert(0, "instrument_id", instrument_id)
    df["ex_date"] = pd.to_datetime(df["ex_date"]).astype("datetime64[ns]")
    df["value"] = df["value"].astype("float64")
    df["vendor"] = vendor
    df["recorded_at"] = pd.Timestamp(now())
    return df


class FakeMacroAdapter:
    """Serves pre-built macro series (series_id -> date-indexed Series); records calls."""

    name = "fakemacro"
    identifier_key = None

    def __init__(self, series: dict, name: str | None = None):
        from quant.data.adapters.base import FieldMap, RateLimit

        if name:
            self.name = name
        self.series = series
        self.calls: list[tuple[list[str], object, object]] = []
        self.rate_limit = RateLimit()
        self.field_map = FieldMap()

    def capabilities(self):
        from quant.data.adapters.base import Capability

        return frozenset({Capability.MACRO})

    def get_macro_series(self, series_ids, start, end):
        import pandas as pd

        from quant.data.adapters.base import AdapterResult
        from quant.data.adapters.fred import _empty_macro, macro_frame

        self.calls.append((list(series_ids), start, end))
        frames, errors = [], {}
        for sid in series_ids:
            s = self.series.get(sid)
            if s is None:
                errors[sid] = "no fixture"
                continue
            window = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
            if window.empty:
                errors[sid] = "no data in range"
                continue
            frames.append(macro_frame(sid, pd.Series(window.index), pd.Series(window.to_numpy()), self.name))
        frame = pd.concat(frames, ignore_index=True) if frames else _empty_macro()
        return AdapterResult(
            vendor=self.name,
            frame=frame,
            errors=errors,
            requested=list(series_ids),
            params={"start": str(start), "end": str(end)},
        )

    def get_bars(self, instruments, start, end, freq="1d"):
        raise NotImplementedError

    def get_corporate_actions(self, instruments, start, end):
        raise NotImplementedError

    def get_fundamentals(self, instruments):
        raise NotImplementedError

    def get_earnings(self, instruments):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


# --------------------------------------------------------------------------- S3: backtest helpers


def zero_costs(**overrides):
    """A CostModel with every cost switched off (tests pin analytic results), overridable per field."""
    from quant.backtest.costs import CostModel

    base = dict(
        half_spread_bps=0.0,
        slippage_bps=0.0,
        fx_cost_bps=0.0,
        borrow_bps_annual=0.0,
        apply_transaction_taxes=False,
    )
    base.update(overrides)
    return CostModel(**base)


class ScheduleStrategy:
    """Emit orders on given session indexes: schedule = {index: [(instrument_id, side, qty, kw)]}."""

    name = "schedule"
    warmup = 0

    def __init__(self, instruments, schedule, weights_at=None):
        self.instruments = list(instruments)
        self.params = {"schedule": {k: [(a, b, c) for a, b, c, *_ in v] for k, v in schedule.items()}}
        self.schedule = schedule
        self.weights_at = weights_at or {}

    def on_start(self, instruments):
        return None

    def on_session(self, ctx):
        if ctx.session_index in self.weights_at:
            return self.weights_at[ctx.session_index]
        orders = []
        for iid, side, qty, *rest in self.schedule.get(ctx.session_index, []):
            kw = rest[0] if rest else {}
            kind = kw.pop("type", "market")
            if kind == "market":
                orders.append(ctx.market_order(iid, side, qty, **kw))
            elif kind == "limit":
                orders.append(ctx.limit_order(iid, side, qty, kw.pop("limit_price"), **kw))
            elif kind == "stop":
                orders.append(ctx.stop_order(iid, side, qty, kw.pop("stop_price"), **kw))
        return orders or None

    def describe(self):
        return {"name": self.name, "instruments": self.instruments, "params": {}}


# --------------------------------------------------------------------------- S7: fundamentals helpers


def yf_statement_frames(scale: float = 1.0):
    """yfinance-style annual + quarterly statement frames (rows = line items, columns = period ends)."""
    import pandas as pd

    annual_cols = pd.to_datetime(["2023-12-31", "2022-12-31", "2021-12-31"])
    income = (
        pd.DataFrame(
            {
                "Total Revenue": [1000.0, 900.0, 800.0],
                "Cost Of Revenue": [600.0, 550.0, 500.0],
                "Operating Income": [200.0, 170.0, 150.0],
                "Interest Expense": [10.0, 10.0, 10.0],
                "Pretax Income": [190.0, 160.0, 140.0],
                "Tax Provision": [40.0, 32.0, 28.0],
                "Net Income": [150.0, 128.0, 112.0],
                "Diluted EPS": [1.5, 1.28, 1.12],
                "Diluted Average Shares": [100.0, 100.0, 100.0],
            },
            index=annual_cols,
        ).T
        * scale
    )
    income.loc["Diluted EPS"] /= scale
    income.loc["Diluted Average Shares"] /= scale
    balance = (
        pd.DataFrame(
            {
                "Total Assets": [2000.0, 1800.0, 1700.0],
                "Current Assets": [600.0, 550.0, 500.0],
                "Cash And Cash Equivalents": [200.0, 150.0, 100.0],
                "Current Liabilities": [300.0, 280.0, 260.0],
                "Total Debt": [400.0, 420.0, 450.0],
                "Stockholders Equity": [1200.0, 1100.0, 1000.0],
                "Ordinary Shares Number": [100.0, 100.0, 100.0],
            },
            index=annual_cols,
        ).T
        * scale
    )
    balance.loc["Ordinary Shares Number"] /= scale
    cashflow = (
        pd.DataFrame(
            {
                "Operating Cash Flow": [220.0, 200.0, 180.0],
                "Capital Expenditure": [-70.0, -60.0, -50.0],
                "Depreciation And Amortization": [50.0, 45.0, 40.0],
                "Cash Dividends Paid": [-30.0, -25.0, -20.0],
            },
            index=annual_cols,
        ).T
        * scale
    )
    q_cols = pd.to_datetime(["2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30"])
    income_q = (
        pd.DataFrame(
            {
                "Total Revenue": [280.0, 260.0, 255.0, 250.0, 245.0],
                "Net Income": [45.0, 40.0, 38.0, 37.0, 36.0],
                "Operating Income": [60.0, 52.0, 50.0, 49.0, 48.0],
            },
            index=q_cols,
        ).T
        * scale
    )
    balance_q = (
        pd.DataFrame(
            {
                "Total Assets": [2150.0, 2100.0, 2000.0, 1950.0, 1900.0],
                "Stockholders Equity": [1260.0, 1230.0, 1200.0, 1170.0, 1140.0],
                "Total Debt": [390.0, 395.0, 400.0, 405.0, 410.0],
                "Cash And Cash Equivalents": [210.0, 205.0, 200.0, 190.0, 180.0],
                "Ordinary Shares Number": [100.0] * 5,
            },
            index=q_cols,
        ).T
        * scale
    )
    balance_q.loc["Ordinary Shares Number"] /= scale
    cashflow_q = (
        pd.DataFrame(
            {
                "Operating Cash Flow": [60.0, 55.0, 56.0, 54.0, 52.0],
                "Capital Expenditure": [-20.0, -18.0, -17.0, -16.0, -15.0],
                "Depreciation And Amortization": [13.0, 13.0, 12.0, 12.0, 12.0],
            },
            index=q_cols,
        ).T
        * scale
    )
    earnings_dates = pd.DataFrame(
        {
            "EPS Estimate": [0.40, 0.38, 0.36, 0.35, 0.44],
            "Reported EPS": [0.45, 0.40, 0.38, 0.37, None],
            "Surprise(%)": [12.5, 5.3, 5.6, 5.7, None],
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-07-25", "2024-04-25", "2024-01-25", "2023-10-26", "2024-10-24"]),
            name="Earnings Date",
        ).tz_localize("America/New_York"),
    )
    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "income_q": income_q,
        "balance_q": balance_q,
        "cashflow_q": cashflow_q,
        "earnings_dates": earnings_dates,
        "info": {"sector": "Technology"},
    }


class FakeFundamentalsAdapter:
    """Serves statements and earnings from a {symbol: frames} map via the real yfinance normaliser."""

    name = "fakefund"
    identifier_key = "yahoo"

    def __init__(self, frames_by_symbol: dict):
        from quant.data.adapters.base import FieldMap, RateLimit

        self.frames = frames_by_symbol
        self.calls: list[str] = []
        self.rate_limit = RateLimit()
        self.field_map = FieldMap()

    def capabilities(self):
        from quant.data.adapters.base import Capability

        return frozenset({Capability.FUNDAMENTALS, Capability.EARNINGS})

    def _inner(self):
        from quant.data.adapters.yfinance_adapter import YFinanceAdapter

        def fn(symbol):
            self.calls.append(symbol)
            if symbol not in self.frames:
                raise KeyError(symbol)
            return self.frames[symbol]

        a = YFinanceAdapter(history_fn=lambda *a: None, fundamentals_fn=fn)
        a.name = self.name
        return a

    def get_fundamentals(self, instruments):
        return self._inner().get_fundamentals(instruments)

    def get_earnings(self, instruments):
        return self._inner().get_earnings(instruments)

    def get_bars(self, instruments, start, end, freq="1d"):
        raise NotImplementedError

    def get_corporate_actions(self, instruments, start, end):
        raise NotImplementedError

    def get_macro_series(self, series_ids, start, end):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None
