from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import actions_frame, make_bars

from quant.data.corporate_actions import adjust_bars, adjustment_factors, dividends, split_ratios
from quant.data.store import DuckDBStore

S, E = date(2024, 6, 3), date(2024, 6, 28)
EX = pd.Timestamp("2024-06-17")


@pytest.fixture
def flat(catalog):
    """Constant-price series (vol=0) so adjustment factors are exact."""
    return make_bars(catalog["XNAS_AAPL"], S, E, vol=0.0, drift=0.0, start_price=100.0)


def test_dividend_adjusts_only_bars_before_the_ex_date(flat):
    acts = actions_frame("XNAS_AAPL", [(EX, "dividend", 2.0, "USD")])
    f = adjustment_factors(flat, acts, state="raw")
    before = flat["session_date"] < EX
    assert np.allclose(f[before], 0.98) and np.allclose(f[~before], 1.0)
    assert before.sum() == 10 and (~before).sum() == 9  # Juneteenth (June 19) closes NYSE


def test_as_of_before_the_ex_date_applies_nothing(flat):
    acts = actions_frame("XNAS_AAPL", [(EX, "dividend", 2.0, "USD")])
    assert np.allclose(adjustment_factors(flat, acts, state="raw", as_of=date(2024, 6, 14)), 1.0)
    assert not np.allclose(adjustment_factors(flat, acts, state="raw", as_of=date(2024, 6, 17)), 1.0)


def test_splits_depend_on_the_stored_adjustment_state(flat):
    acts = actions_frame("XNAS_AAPL", [(EX, "split", 2.0, None)])
    before = flat["session_date"] < EX
    raw = adjustment_factors(flat, acts, state="raw")
    assert np.allclose(raw[before], 0.5) and np.allclose(raw[~before], 1.0)
    assert np.allclose(adjustment_factors(flat, acts, state="split_adjusted"), 1.0)  # vendor already did it
    assert np.allclose(adjustment_factors(flat, acts, state="fully_adjusted"), 1.0)


def test_dividend_and_split_compound(flat):
    acts = actions_frame("XNAS_AAPL", [(EX, "split", 2.0, None), (EX, "dividend", 2.0, "USD")])
    f = adjustment_factors(flat, acts, state="raw")
    assert np.allclose(f[flat["session_date"] < EX], 0.49)


def test_nonsense_actions_are_skipped(flat):
    acts = actions_frame(
        "XNAS_AAPL",
        [
            (EX, "dividend", 150.0, "USD"),  # larger than the price: cannot be right
            (EX, "split", 0.0, None),  # zero ratio
            (pd.Timestamp("2024-05-01"), "split", 2.0, None),  # before the first bar: nothing to adjust
        ],
    )
    assert np.allclose(adjustment_factors(flat, acts, state="raw"), 1.0)


def test_adjust_bars_handles_many_instruments_and_states(catalog):
    a = make_bars(catalog["XNAS_AAPL"], S, E, vol=0.0, drift=0.0, start_price=100.0, adjustment_state="raw")
    v = make_bars(
        catalog["XLON_VOD"], S, E, vol=0.0, drift=0.0, start_price=1.0, adjustment_state="split_adjusted"
    )
    acts = pd.concat(
        [
            actions_frame("XNAS_AAPL", [(EX, "split", 2.0, None)]),
            actions_frame("XLON_VOD", [(EX, "split", 2.0, None), (EX, "dividend", 0.05, "GBP")]),
        ],
        ignore_index=True,
    )
    out = adjust_bars(pd.concat([v, a]), acts)
    assert out["instrument_id"].tolist() == sorted(out["instrument_id"].tolist())
    aapl = out[out["instrument_id"] == "XNAS_AAPL"]
    vod = out[out["instrument_id"] == "XLON_VOD"]
    assert np.allclose(aapl["adj_close"], aapl["close"] * aapl["adj_factor"])
    assert np.allclose(aapl[aapl["session_date"] < EX]["adj_factor"], 0.5)
    assert np.allclose(vod[vod["session_date"] < EX]["adj_factor"], 0.95)  # dividend only; split ignored
    for col in ("adj_open", "adj_high", "adj_low", "adj_close", "adj_factor"):
        assert col in out.columns
    empty = adjust_bars(a.iloc[0:0], acts)
    assert empty.empty and "adj_close" in empty.columns


def test_helpers(catalog):
    acts = actions_frame("XNAS_AAPL", [(EX, "split", 4.0, None), (EX, "dividend", 0.25, "USD")])
    ratios = split_ratios(acts, "XNAS_AAPL")
    assert ratios.index[0] == EX and ratios.iloc[0] == 4.0
    assert split_ratios(acts, "XLON_VOD").empty
    divs = dividends(acts, "XNAS_AAPL")
    assert divs["amount"].tolist() == [0.25] and divs["currency"].tolist() == ["USD"]


def test_actions_round_trip_through_duckdb(tmp_path):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    acts = actions_frame("XNAS_AAPL", [(EX, "split", 2.0, None), (EX, "dividend", 1.0, "USD")])
    assert db.upsert_corporate_actions(acts) == 2
    assert db.upsert_corporate_actions(acts) == 2  # primary key: no duplicates
    back = db.load_corporate_actions(["XNAS_AAPL", "XLON_VOD"])
    assert len(back) == 2 and set(back["action_type"]) == {"split", "dividend"}
    assert back[back["action_type"] == "split"]["currency"].iloc[0] is None
    assert str(back["ex_date"].dtype) == "datetime64[ns]"
    assert db.load_corporate_actions(["XLON_VOD"]).empty
    assert db.load_corporate_actions([]).empty
    too_early = acts["recorded_at"].iloc[0] - pd.Timedelta(days=1)
    assert db.load_corporate_actions(["XNAS_AAPL"], known_by=too_early).empty
    assert len(db.load_corporate_actions(["XNAS_AAPL"], vendor="synthetic")) == 2
    assert db.load_corporate_actions(["XNAS_AAPL"], vendor="other").empty
