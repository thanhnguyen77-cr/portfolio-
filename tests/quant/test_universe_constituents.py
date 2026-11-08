from __future__ import annotations

import hashlib

import pytest
from conftest import FIXTURES

from quant.core.errors import UniverseError
from quant.universe.constituents import (
    UNIVERSES,
    WikipediaSource,
    load_snapshot,
    members_to_instruments,
    normalise_uk_symbol,
    normalise_us_symbol,
    parse_constituents,
    parse_sp500_changes,
)

SP = (FIXTURES / "wiki_sp500.html").read_text()
NDX = (FIXTURES / "wiki_ndx100.html").read_text()
FTSE = (FIXTURES / "wiki_ftse100.html").read_text()
SMALL = (3, 20)


def test_symbol_normalisation():
    assert normalise_us_symbol("BRK.B") == ("BRK-B", "BRK-B")
    assert normalise_uk_symbol("BT.A") == ("BT-A", "BT-A.L")
    assert normalise_uk_symbol("BP.") == ("BP", "BP.L")
    assert normalise_uk_symbol(" rr. ") == ("RR", "RR.L")


def test_sp500_members_and_changes_parse():
    members, sha = parse_constituents(SP, UNIVERSES["sp500"], expected_rows=SMALL)
    assert sha == hashlib.sha256(SP.encode()).hexdigest()
    assert members["symbol"].tolist() == ["AAPL", "MSFT", "BRK-B", "JPM", "XOM", "META"]
    assert members["yahoo"].tolist()[2] == "BRK-B" and (members["market"] == "XNYS").all()
    assert members.loc[0, "sector"] == "Information Technology"
    changes = parse_sp500_changes(SP)
    assert changes["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-05", "2024-03-18", "2024-06-20"]
    assert changes["added"].tolist() == ["JPM", "XOM", "META"] and changes["removed"].tolist() == [
        "OLD2",
        "OLD1",
        "FB",
    ]
    assert "Ticker change" in changes["reason"].iloc[-1]


def test_ndx_and_ftse_parse():
    ndx, _ = parse_constituents(NDX, UNIVERSES["ndx100"], expected_rows=SMALL)
    assert ndx["symbol"].tolist() == ["AAPL", "MSFT", "NVDA", "AMZN"] and (ndx["market"] == "XNAS").all()
    ftse, _ = parse_constituents(FTSE, UNIVERSES["ftse100"], expected_rows=SMALL)
    assert ftse["symbol"].tolist() == ["VOD", "BP", "BT-A", "SHEL", "RR"]
    assert ftse["yahoo"].tolist() == ["VOD.L", "BP.L", "BT-A.L", "SHEL.L", "RR.L"]
    assert (
        ftse["sector"].iloc[0] == "Telecommunications Service Providers"
    )  # footnote marker stripped from header


def test_shape_drift_is_an_error():
    with pytest.raises(UniverseError, match="shape drifted"):
        parse_constituents(FTSE, UNIVERSES["ftse100"], expected_rows=(50, 60))
    broken = FTSE.replace("<th>Ticker</th>", "<th>Code</th>").replace("Ticker", "Code")
    with pytest.raises(UniverseError, match="shape drifted"):
        parse_constituents(broken, UNIVERSES["ftse100"], expected_rows=SMALL)
    with pytest.raises(UniverseError, match="changes table not found"):
        parse_sp500_changes(FTSE)
    with pytest.raises(UniverseError, match="no HTML tables"):
        parse_constituents("<html><body>nothing</body></html>", UNIVERSES["sp500"], expected_rows=SMALL)


def test_load_snapshot_through_the_http_seam():
    src = WikipediaSource()
    calls: list[str] = []

    class _Resp:
        def __init__(self, text):
            self.text = text

    def _http(method, url, **kw):
        calls.append(url)
        return _Resp(SP if "S%26P" in url else FTSE)

    src._http = _http
    snap = load_snapshot("sp500", src, expected_rows=SMALL)
    assert snap.changes is not None and len(snap.members) == 6 and calls == [UNIVERSES["sp500"].url]
    ftse = load_snapshot("ftse100", src, expected_rows=SMALL)
    assert ftse.changes is None and any("current_only" in n for n in ftse.notes)
    with pytest.raises(UniverseError, match="unknown universe"):
        load_snapshot("nope", src)


def test_members_become_instruments_reusing_the_catalogue(catalog):
    members, _ = parse_constituents(SP, UNIVERSES["sp500"], expected_rows=SMALL)
    insts = {i.symbol: i for i in members_to_instruments(members, UNIVERSES["sp500"], catalog)}
    assert insts["AAPL"] is catalog["XNAS_AAPL"]  # reused, keeps its Nasdaq id
    assert insts["BRK-B"].instrument_id == "XNYS_BRK_B" and insts["BRK-B"].identifiers["yahoo"] == "BRK-B"
    ftse, _ = parse_constituents(FTSE, UNIVERSES["ftse100"], expected_rows=SMALL)
    uk = {i.symbol: i for i in members_to_instruments(ftse, UNIVERSES["ftse100"], catalog)}
    assert uk["VOD"] is catalog["XLON_VOD"]
    bt = uk["BT-A"]
    assert bt.instrument_id == "XLON_BT_A" and bt.price_scale == 0.01 and bt.currency == "GBP"
    assert bt.stamp_duty_class == "unknown" and bt.identifiers["yahoo"] == "BT-A.L"
