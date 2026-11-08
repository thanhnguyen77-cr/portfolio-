from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES, FakeFundamentalsAdapter, yf_statement_frames

from quant.config.settings import load_settings
from quant.core.errors import AdapterError, AdapterNotConfigured
from quant.data.adapters.sec_edgar import SECEdgarAdapter
from quant.data.adapters.yfinance_adapter import YFinanceAdapter
from quant.data.service import DataService
from quant.data.store import DuckDBStore
from quant.fundamental import (
    StatementSet,
    comparable_valuation,
    compute_ratios,
    dcf_sensitivity,
    dcf_valuation,
    earnings_summary,
    from_edgar_companyfacts,
    from_yfinance,
    peer_multiples,
    post_earnings_returns,
    score_cross_section,
    surprises,
    wacc,
)


@pytest.fixture
def aapl_rows(catalog):
    return from_yfinance(catalog["XNAS_AAPL"], yf_statement_frames())


def test_yfinance_normalisation_maps_items_and_estimates_filing_dates(aapl_rows):
    rows = aapl_rows
    assert set(rows["item"]) >= {
        "revenue",
        "net_income",
        "total_assets",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "gross_profit",
        "ebit",
        "ebitda",
        "total_debt",
        "eps_diluted",
        "shares_outstanding",
    }
    fy23 = rows[(rows["period_end"] == date(2023, 12, 31)) & (rows["period_type"] == "annual")].set_index(
        "item"
    )["value"]
    assert fy23["revenue"] == 1000.0 and fy23["gross_profit"] == 400.0 and fy23["ebitda"] == 250.0
    assert fy23["free_cash_flow"] == pytest.approx(150.0)
    filed = rows[(rows["period_end"] == date(2023, 12, 31)) & (rows["period_type"] == "annual")][
        "filed_at"
    ].unique()
    assert list(filed) == [date(2024, 3, 15)]  # 75-day lag, flagged as estimated
    assert rows["filed_at_estimated"].all()
    q = rows[(rows["period_type"] == "quarterly") & (rows["item"] == "revenue")]
    assert len(q) == 5 and (q["filed_at"] - q["period_end"]).map(lambda d: d.days).eq(45).all()


def test_statement_set_point_in_time_ttm_and_growth(aapl_rows):
    s = StatementSet.from_rows(aapl_rows)
    assert s.latest("revenue") == 1000.0 and s.latest_period_end("annual") == date(2023, 12, 31)
    before_10k = s.as_of(date(2024, 2, 1))  # FY2023 estimated filing is 2024-03-15
    assert before_10k.latest("revenue") == 900.0 and before_10k.latest_period_end("annual") == date(
        2022, 12, 31
    )
    assert s.as_of(date(2024, 3, 15)).latest("revenue") == 1000.0
    assert s.ttm("revenue") == pytest.approx(280 + 260 + 255 + 250)  # last four quarters
    assert s.ttm("total_assets") == 2150.0  # stock item: latest quarter
    assert s.as_of(date(2024, 5, 1)).ttm("revenue") == 1000.0  # only three quarters filed: annual fallback
    assert s.as_of(date(2024, 5, 20)).ttm("revenue") == pytest.approx(260 + 255 + 250 + 245)  # Q1 filed 05-15
    assert s.yoy_growth("revenue") == pytest.approx(1000 / 900 - 1)
    assert s.filing_lag_estimated() and "revenue" in s.items()
    assert (
        StatementSet.from_rows(pd.DataFrame()).items() == []
        and StatementSet.from_rows(pd.DataFrame()).ttm("revenue") is None
    )


def test_edgar_companyfacts_true_filing_dates_and_ytd_skipping(catalog):
    facts = json.loads((FIXTURES / "edgar_companyfacts_sample.json").read_text())
    rows = from_edgar_companyfacts(catalog["XNAS_AAPL"], facts)
    assert not rows["filed_at_estimated"].any()
    rev = rows[rows["item"] == "revenue"]
    assert sorted(rev["period_type"]) == ["annual", "quarterly", "quarterly"]
    q2 = rev[rev["period_end"] == date(2024, 6, 30)]
    assert len(q2) == 1 and q2["value"].iloc[0] == 280.0  # the 540 year-to-date figure was skipped
    assert rev[rev["period_type"] == "annual"]["filed_at"].iloc[0] == date(2024, 2, 15)
    assert rows[rows["item"] == "total_debt"]["value"].iloc[0] == 350.0  # derived from long + short term
    assert rows[rows["item"] == "shares_outstanding"]["value"].iloc[0] == 100.0
    assert "gross_profit" in set(rows["item"]) and "ebit" in set(rows["item"])
    s = StatementSet.from_rows(rows)
    assert (
        s.as_of(date(2024, 2, 14)).latest("revenue") is None
        and s.as_of(date(2024, 2, 15)).latest("revenue") == 1000.0
    )


def test_duckdb_round_trip_with_point_in_time_filter(tmp_path, aapl_rows):
    db = DuckDBStore(tmp_path / "q.duckdb")
    db.migrate()
    assert db.upsert_fundamentals(aapl_rows) == len(aapl_rows)
    assert db.upsert_fundamentals(aapl_rows) == len(aapl_rows)
    back = db.load_fundamentals(["XNAS_AAPL"])
    assert len(back) == len(aapl_rows) and set(back["item"]) == set(aapl_rows["item"])
    early = db.load_fundamentals(["XNAS_AAPL"], as_of=date(2024, 2, 1))
    assert early["period_end"].max() < date(2024, 1, 1) or early[early["period_type"] == "annual"][
        "period_end"
    ].max() == date(2022, 12, 31)
    assert db.load_fundamentals([]).empty and db.load_fundamentals(["XLON_VOD"]).empty


def test_ratios_match_hand_calculations(aapl_rows):
    s = StatementSet.from_rows(aapl_rows)
    r = compute_ratios(s, price=30.0)
    rev_ttm = 280 + 260 + 255 + 250
    ni_ttm = 45 + 40 + 38 + 37
    assert r["revenue_ttm"] == pytest.approx(rev_ttm) and r["net_income_ttm"] == pytest.approx(ni_ttm)
    assert r["net_margin"] == pytest.approx(ni_ttm / rev_ttm)
    assert r["gross_margin"] == pytest.approx(400 / 1000)  # gross profit only annual -> falls back to FY2023
    assert r["roe"] == pytest.approx(ni_ttm / 1260.0) and r["debt_to_equity"] == pytest.approx(390 / 1260)
    assert r["tax_rate"] == pytest.approx(40 / 190) and r["assumptions"] == []
    ebit_ttm = 60 + 52 + 50 + 49
    nopat = ebit_ttm * (1 - 40 / 190)
    assert r["roic"] == pytest.approx(nopat / (1260 + 390 - 210))
    assert r["market_cap"] == 3000.0 and r["enterprise_value"] == pytest.approx(3000 + 390 - 210)
    assert r["pe"] == pytest.approx(30.0 / (ni_ttm / 100)) and r["earnings_yield"] == pytest.approx(
        ni_ttm / 3000
    )
    assert r["fcf_yield"] == pytest.approx((60 + 55 + 56 + 54 - (20 + 18 + 17 + 16)) / 3000)
    assert r["revenue_growth"] == pytest.approx(1000 / 900 - 1) and r["current_ratio"] == pytest.approx(
        600 / 300
    )
    assert r["interest_coverage"] == pytest.approx(200 / 10)  # annual pair: interest expense is annual only
    assert r["operating_margin"] == pytest.approx((60 + 52 + 50 + 49) / rev_ttm)  # both quarterly -> TTM pair
    assert r["accruals"] == pytest.approx((ni_ttm - (60 + 55 + 56 + 54)) / 2150)
    no_price = compute_ratios(s)
    assert "pe" not in no_price and no_price["roa"] == pytest.approx(ni_ttm / 2150)
    pit = compute_ratios(s, price=30.0, as_of=date(2024, 2, 1))
    assert pit["latest_annual_period"] == date(2022, 12, 31)


def test_dcf_matches_a_spreadsheet_and_is_monotone():
    flows = [100 * 1.05**k for k in range(1, 6)]
    pv = sum(f / 1.08**t for t, f in enumerate(flows, start=1))
    tv = flows[-1] * 1.02 / (0.08 - 0.02) / 1.08**5
    res = dcf_valuation(
        100.0, 0.05, years=5, terminal_growth=0.02, discount_rate=0.08, net_debt=200.0, shares=50.0
    )
    assert res.pv_explicit == pytest.approx(pv) and res.pv_terminal == pytest.approx(tv)
    assert res.enterprise_value == pytest.approx(pv + tv) and res.equity_value == pytest.approx(pv + tv - 200)
    assert res.per_share == pytest.approx((pv + tv - 200) / 50) and 0 < res.terminal_share < 1
    assert res.cash_flows == pytest.approx(flows) and res.as_dict()["assumption_years"] == 5
    path = dcf_valuation(100.0, [0.10, 0.05, 0.0], terminal_growth=0.02, discount_rate=0.09)
    assert path.cash_flows == pytest.approx([110.0, 115.5, 115.5])
    mid = dcf_valuation(100.0, 0.05, discount_rate=0.08, mid_year=True)
    assert mid.enterprise_value > res.enterprise_value - 0  # earlier cash flows discount less
    with pytest.raises(ValueError, match="exceed"):
        dcf_valuation(100.0, 0.05, terminal_growth=0.09, discount_rate=0.08)
    assert wacc(0.04, 0.05, 1.2, 0.05, 0.25, 0.3) == pytest.approx(0.7 * 0.10 + 0.3 * 0.05 * 0.75)
    with pytest.raises(ValueError):
        wacc(0.04, 0.05, 1.0, 0.05, 0.2, 1.5)
    grid = dcf_sensitivity(
        100.0, 0.05, discount_rates=[0.07, 0.08, 0.09], terminal_growths=[0.01, 0.02, 0.03], shares=50.0
    )
    assert (
        grid.shape == (3, 3) and (grid.iloc[0] > grid.iloc[1]).all() and (grid.iloc[1] > grid.iloc[2]).all()
    )
    assert (grid.iloc[:, 2] > grid.iloc[:, 0]).all()
    assert np.isnan(dcf_sensitivity(100.0, 0.05, discount_rates=[0.02], terminal_growths=[0.03]).iloc[0, 0])


def test_comparable_valuation():
    peers = peer_multiples(
        {
            "p1": {"pe": 10.0, "ev_ebitda": 8.0, "pb": 1.5},
            "p2": {"pe": 12.0, "ev_ebitda": 9.0, "pb": 2.0},
            "p3": {"pe": 14.0, "ev_ebitda": 10.0, "pb": None},
        }
    )
    assert peers.shape == (3, 5) and np.isnan(peers.loc["p3", "pb"])
    target = {
        "net_income": 50.0,
        "ebitda": 80.0,
        "total_equity": 300.0,
        "revenue": 500.0,
        "net_debt": 100.0,
        "shares": 100.0,
    }
    out = comparable_valuation(target, peers).set_index("multiple")
    assert out.loc["pe", "implied_equity_value"] == 600.0 and out.loc["pe", "implied_per_share"] == 6.0
    assert out.loc["ev_ebitda", "implied_equity_value"] == pytest.approx(9.0 * 80 - 100)
    assert out.loc["pb", "peers_used"] == 2 and out.loc["pb", "implied_equity_value"] == pytest.approx(
        1.75 * 300
    )
    assert "ps" not in out.index and "ev_sales" not in out.index  # no peer data for those
    assert out.loc["median_of_multiples", "implied_per_share"] == pytest.approx(np.median([6.0, 6.2, 5.25]))
    with pytest.raises(ValueError):
        comparable_valuation(target, peers, statistic="mode")
    assert comparable_valuation({"net_income": -5.0}, peers).empty


def test_scores_rank_the_planted_best_company_first():
    ratios = pd.DataFrame(
        {
            "roic": [0.25, 0.10, 0.02],
            "roe": [0.30, 0.12, 0.01],
            "gross_margin": [0.6, 0.4, 0.2],
            "fcf_margin": [0.2, 0.1, -0.05],
            "accruals": [-0.02, 0.01, 0.08],
            "debt_to_equity": [0.2, 0.8, 2.5],
            "earnings_yield": [0.08, 0.05, 0.02],
            "fcf_yield": [0.07, 0.04, -0.01],
            "pb": [2.0, 3.0, 6.0],
            "ev_ebitda": [8.0, 12.0, 25.0],
            "ps": [1.0, 2.0, 5.0],
            "revenue_growth": [0.15, 0.05, -0.1],
            "eps_growth": [0.2, 0.0, -0.3],
            "net_income_growth": [0.18, 0.02, -0.4],
        },
        index=["GREAT", "OK", "POOR"],
    )
    scores = score_cross_section(ratios)
    assert scores.index.tolist() == ["GREAT", "OK", "POOR"]
    assert scores.loc["GREAT", "quality_score"] > scores.loc["POOR", "quality_score"]
    assert scores.loc["GREAT", "value_score"] > scores.loc["POOR", "value_score"]
    assert (
        (scores["quality_n"] == 6).all()
        and (scores["value_n"] == 5).all()
        and (scores["growth_n"] == 3).all()
    )
    partial = score_cross_section(ratios[["roic", "pb"]])
    assert (
        (partial["growth_n"] == 0).all()
        and partial["growth_score"].isna().all()
        and partial["composite"].notna().all()
    )


def test_earnings_surprises_and_reactions():
    df = pd.DataFrame(
        {
            "report_date": pd.to_datetime(
                [
                    "2023-01-25",
                    "2023-04-26",
                    "2023-07-26",
                    "2023-10-26",
                    "2024-01-25",
                    "2024-04-25",
                    "2024-07-25",
                ]
            ).date,
            "eps_estimate": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "eps_actual": [1.1, 1.15, 1.1, 1.4, 1.5, 1.45, 1.9],
        }
    )
    s = surprises(df, sue_window=4)
    assert s["surprise"].tolist() == pytest.approx([0.1, 0.05, -0.1, 0.1, 0.1, -0.05, 0.3])
    assert s["beat"].tolist() == [True, True, False, True, True, False, True]
    assert np.isnan(s["sue"].iloc[0]) and np.isfinite(s["sue"].iloc[-1]) and s["sue"].iloc[-1] > 1
    summary = earnings_summary(s)
    assert (
        summary["reports"] == 7 and summary["beat_rate"] == pytest.approx(5 / 7) and summary["last_sue"] > 1
    )
    assert (
        earnings_summary(pd.DataFrame(columns=["report_date", "eps_estimate", "eps_actual"]))["reports"] == 0
    )
    idx = pd.bdate_range("2024-07-15", periods=40)
    prices = pd.Series(np.linspace(100, 139, 40), index=idx)
    reaction = post_earnings_returns(prices, [date(2024, 7, 25)], windows=(1, 5))
    pos = idx.get_loc(pd.Timestamp("2024-07-25"))
    assert reaction["ret_1d"].iloc[0] == pytest.approx(prices.iloc[pos] / prices.iloc[pos - 1] - 1)
    assert reaction["ret_5d"].iloc[0] == pytest.approx(prices.iloc[pos + 4] / prices.iloc[pos - 1] - 1)
    assert post_earnings_returns(prices, [date(2020, 1, 1)]).empty


def test_yfinance_adapter_fundamentals_and_earnings_seam(catalog):
    frames = yf_statement_frames()
    calls: list[str] = []

    def fn(symbol):
        calls.append(symbol)
        if symbol != "AAPL":
            raise RuntimeError("no such symbol")
        return frames

    adapter = YFinanceAdapter(history_fn=lambda *a: None, fundamentals_fn=fn)
    res = adapter.get_fundamentals([catalog["XNAS_AAPL"], catalog["XLON_VOD"]])
    assert res.ok == ["XNAS_AAPL"] and "RuntimeError" in res.errors["XLON_VOD"] and calls == ["AAPL", "VOD.L"]
    assert res.extras["info"]["XNAS_AAPL"] == {"sector": "Technology"}
    earn = adapter.get_earnings([catalog["XNAS_AAPL"]])
    e = earn.frame
    assert (
        len(e) == 5
        and e["report_date"].iloc[0] == pd.Timestamp("2023-10-26")
        and e["eps_actual"].isna().sum() == 1
    )
    assert (e["currency"] == "USD").all() and e["report_date"].is_monotonic_increasing


def test_edgar_adapter_requires_contact_and_parses(catalog):
    tickers = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK.B", "title": "Berkshire"},
    }
    facts = json.loads((FIXTURES / "edgar_companyfacts_sample.json").read_text())
    with pytest.raises(AdapterNotConfigured, match="QUANT_SEC_USER_AGENT"):
        SECEdgarAdapter().get_fundamentals([catalog["XNAS_AAPL"]])
    adapter = SECEdgarAdapter(user_agent="Test User test@example.com")
    urls: list[str] = []

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    def _http(method, url, **kw):
        urls.append(url)
        return _Resp(tickers if url.endswith("company_tickers.json") else facts)

    adapter._http = _http
    assert SECEdgarAdapter.parse_tickers(tickers) == {"AAPL": 320193, "BRK-B": 1067983}
    res = adapter.get_fundamentals([catalog["XNAS_AAPL"], catalog["XNYS_JPM"]])
    assert res.ok == ["XNAS_AAPL"] and "no CIK" in res.errors["XNYS_JPM"]
    assert any("CIK0000320193.json" in u for u in urls) and adapter.user_agent.startswith("Test User")
    assert not res.frame["filed_at_estimated"].any()
    with pytest.raises(AdapterError):
        adapter.cik_for(catalog["XNYS_JPM"])


def test_service_fundamentals_and_earnings_cache(tmp_path, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(
        paths={"data_dir": str(tmp_path / "data")}, data={"adapter_priority": ["fakefund"]}
    )
    fake = FakeFundamentalsAdapter({"AAPL": yf_statement_frames(), "VOD.L": yf_statement_frames(scale=0.5)})
    svc = DataService(settings, catalog=catalog, adapters={"fakefund": fake})
    s = svc.get_fundamentals("AAPL")
    assert s.latest("revenue") == 1000.0 and fake.calls == ["AAPL"]
    svc.get_fundamentals("AAPL")  # cached: no second call
    assert fake.calls == ["AAPL"]
    pit = svc.get_fundamentals("AAPL", as_of=date(2024, 2, 1))
    assert pit.latest("revenue") == 900.0
    assert svc.get_fundamentals("XNYS_JPM", refresh=False).rows.empty
    rep = svc.fetch_fundamentals(["VOD.L"])
    assert rep.rows > 0 and svc.get_fundamentals("VOD.L").latest("revenue") == 500.0
    earn = svc.get_earnings("AAPL")
    assert len(earn) == 5 and svc.get_earnings("AAPL", as_of=date(2024, 5, 1))["report_date"].max() == date(
        2024, 4, 25
    )
    ops = svc.lineage.recent()["operation"].tolist()
    assert "fetch_fundamentals" in ops and "fetch_earnings" in ops
