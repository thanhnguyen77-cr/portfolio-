from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars

from quant.core.errors import FXError
from quant.data.fx import FIXING_NOTE, FXRates

DATES = pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-04"])  # Wednesday 3rd missing


@pytest.fixture
def fx():
    gbpusd = pd.Series([1.25, 1.26, 1.28], index=DATES)
    eurusd = pd.Series([1.10, 1.10, 1.12], index=DATES)
    return FXRates({"GBPUSD": gbpusd, "EURUSD": eurusd})


def test_direct_inverse_identity(fx):
    assert fx.rate("GBP", "USD", DATES).tolist() == [1.25, 1.26, 1.28]
    assert np.allclose(fx.rate("USD", "GBP", DATES), [1 / 1.25, 1 / 1.26, 1 / 1.28])
    assert fx.rate("GBP", "GBP", DATES).eq(1.0).all()
    assert fx.pairs == ["EURUSD", "GBPUSD"]


def test_cross_through_usd(fx):
    gbpeur = fx.rate("GBP", "EUR", DATES)
    assert np.allclose(gbpeur, [1.25 / 1.10, 1.26 / 1.10, 1.28 / 1.12])
    assert np.allclose(fx.rate("EUR", "GBP", DATES), 1 / gbpeur)


def test_forward_fill_but_never_backfill(fx):
    r = fx.rate("GBP", "USD", pd.to_datetime(["2024-06-28", "2024-07-03", "2024-07-10"]))
    assert np.isnan(r.iloc[0])  # before the first fix: unknown, not borrowed from the future
    assert r.iloc[1] == 1.26  # Wednesday takes Tuesday's fix
    assert r.iloc[2] == 1.28  # carried forward
    with pytest.raises(FXError, match="2024-06-28"):
        fx.convert(pd.Series([1.0], index=pd.to_datetime(["2024-06-28"])), "GBP", "USD")
    amounts = pd.Series([1.0, 100.0], index=pd.to_datetime(["2024-06-28", "2024-07-03"]))
    allowed = fx.convert(amounts, "GBP", "USD", allow_missing=True)
    assert np.isnan(allowed.iloc[0]) and allowed.iloc[1] == 126.0


def test_latest_and_missing_pair(fx):
    assert fx.latest("GBP", "USD", date(2024, 7, 3)) == 1.26
    with pytest.raises(FXError, match="no FX rate for JPY->CHF"):
        fx.rate("JPY", "CHF", DATES)
    with pytest.raises(FXError, match="on or before 2024-06-01"):
        fx.latest("GBP", "USD", date(2024, 6, 1))


def test_from_bars_uses_fx_instruments_only(catalog):
    bars = pd.concat(
        [
            make_bars(catalog["FX_GBPUSD"], date(2024, 7, 1), date(2024, 7, 12), start_price=1.27, vol=0.002),
            make_bars(catalog["XNAS_AAPL"], date(2024, 7, 1), date(2024, 7, 12)),
        ]
    )
    fx = FXRates.from_bars(bars, catalog)
    assert fx.pairs == ["GBPUSD"]
    sessions = bars[bars["instrument_id"] == "FX_GBPUSD"]["session_date"]
    assert fx.rate("GBP", "USD", sessions).notna().all()
    assert date(2024, 7, 4) in set(sessions.dt.date)  # FX trades on the US holiday


def test_bad_symbol_and_note():
    with pytest.raises(FXError, match="two ISO codes"):
        FXRates({"GBP": pd.Series([1.0], index=DATES[:1])})
    assert "forward-filled" in FIXING_NOTE
