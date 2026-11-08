from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd
import pytest
from conftest import FakeBarsAdapter, make_bars, zero_costs

from quant.backtest.engine import Backtest, BacktestConfig
from quant.backtest.runner import assemble, run_from_service
from quant.config.settings import load_settings
from quant.core.time import frozen_now
from quant.data.service import DataService
from quant.reports.tearsheet import build_tearsheet
from quant.strategies import BuyAndHold, CrossSectionalMomentum, TrendFollowing

S, E = date(2023, 1, 3), date(2024, 6, 28)


@pytest.fixture
def svc(tmp_path, catalog, monkeypatch):
    monkeypatch.chdir(tmp_path)
    series = {
        "XNAS_AAPL": make_bars(catalog["XNAS_AAPL"], date(2021, 6, 1), E, seed=51),
        "XNYS_JPM": make_bars(catalog["XNYS_JPM"], date(2021, 6, 1), E, seed=52),
        "XLON_VOD": make_bars(catalog["XLON_VOD"], date(2021, 6, 1), E, seed=53, start_price=0.8),
        "FX_GBPUSD": make_bars(
            catalog["FX_GBPUSD"], date(2021, 6, 1), E, seed=54, start_price=1.25, vol=0.004
        ),
    }
    settings = load_settings(paths={"data_dir": str(tmp_path / "data")}, data={"adapter_priority": ["fake"]})
    return DataService(settings, catalog=catalog, adapters={"fake": FakeBarsAdapter(series)})


def test_assemble_and_run_from_service_with_universe_caveats(svc):
    uni = svc.universe("demo")
    uni.import_intervals(
        pd.DataFrame(
            {
                "instrument_id": ["XNAS_AAPL", "XNYS_JPM", "XLON_VOD", "XNYS_XOM"],
                "effective_from": ["2020-01-01"] * 4,
            }
        ),
        source="test",
    )
    with frozen_now("2024-06-29T12:00:00Z"):
        parts = assemble(
            svc, ["XNAS_AAPL", "XNYS_JPM", "XLON_VOD", "XNYS_XOM"], S, E, base_currency="USD", universe="demo"
        )
    assert parts["missing"] == ["XNYS_XOM"] and set(parts["instruments"]) == {
        "XNAS_AAPL",
        "XNYS_JPM",
        "XLON_VOD",
    }
    assert parts["fx"] is not None and "GBPUSD" in parts["fx"].pairs
    assert parts["bars"]["session_date"].min().date() < S  # warm-up buffer fetched
    assert any("Data quality grades" in c for c in parts["caveats"])
    assert any("coverage of historical members with price data: 75%" in c for c in parts["caveats"])
    assert any("No price data for ['XNYS_XOM']" in c for c in parts["caveats"])
    cfg = BacktestConfig(start=S, end=E, initial_cash=50_000, cost_model=zero_costs(), fractional_shares=True)
    with frozen_now("2024-06-29T12:00:00Z"):
        res = run_from_service(
            svc,
            TrendFollowing(["XNAS_AAPL", "XNYS_JPM", "XLON_VOD", "XNYS_XOM"], fast=5, slow=20),
            cfg,
            universe="demo",
        )
    assert res.equity["session_date"].iloc[0].date() >= S and len(res.fills) > 0
    assert any("quality: reconstructed" in c for c in res.caveats) and any(
        "XNYS_XOM" in c for c in res.caveats
    )


def test_xs_momentum_picks_the_strongest_names(catalog):
    s, e = date(2022, 1, 3), date(2023, 12, 29)
    drifts = {"XNAS_AAPL": 0.003, "XNAS_MSFT": 0.001, "XNYS_JPM": -0.001, "XNYS_XOM": -0.003}
    bars = pd.concat(
        [
            make_bars(catalog[iid], s, e, seed=i, vol=0.003, drift=d)
            for i, (iid, d) in enumerate(drifts.items())
        ]
    )
    strat = CrossSectionalMomentum(list(drifts), lookback=120, skip=10, top_n=2)
    cfg = BacktestConfig(
        start=s, end=e, initial_cash=100_000, cost_model=zero_costs(), fractional_shares=True
    )
    res = Backtest(cfg, strat, bars, catalog).run()
    held = res.positions[res.positions["session_date"] == res.positions["session_date"].max()]
    assert set(held["instrument_id"]) == {"XNAS_AAPL", "XNAS_MSFT"}
    rebalances = res.orders["created_session"].nunique()
    assert 8 <= rebalances <= 20  # monthly after a ~6-month warm-up over two years
    with pytest.raises(ValueError):
        CrossSectionalMomentum(["XNAS_AAPL"], lookback=10, skip=20)


def test_tearsheet_structure_and_role_based_colouring(catalog, tmp_path):
    bars = pd.concat(
        [make_bars(catalog["XNAS_AAPL"], S, E, seed=61), make_bars(catalog["XNYS_JPM"], S, E, seed=62)]
    )
    bench = bars[bars["instrument_id"] == "XNYS_JPM"].set_index("session_date")["close"]
    cfg = BacktestConfig(
        start=S, end=E, initial_cash=100_000, cost_model=zero_costs(), fractional_shares=True
    )
    res = Backtest(cfg, BuyAndHold(["XNAS_AAPL", "XNYS_JPM"]), bars, catalog, benchmark=bench).run()
    from quant.backtest.analysis import block_bootstrap_sharpe, deflated_sharpe_ratio

    rets = res.equity_series.pct_change().dropna()
    robustness = {
        "bootstrap": block_bootstrap_sharpe(rets, n_boot=50),
        "deflated_sharpe": deflated_sharpe_ratio(rets, 3),
    }
    html = build_tearsheet(res, robustness=robustness, path=tmp_path / "tearsheet.html")
    assert (tmp_path / "tearsheet.html").exists()
    assert "<title>Tearsheet: buy_and_hold</title>" in html and "Caveats" in html
    specs = re.findall(r'<script type="application/json" data-chart="([^"]+)">(.*?)</script>', html, re.S)
    assert len(specs) == 4  # equity, drawdown, yearly returns, exposure
    for chart_id, raw in specs:
        spec = json.loads(raw)
        for trace in spec["data"]:
            assert "meta" in trace and trace["meta"]["role"] in {"series", "polarity"}
            assert "#" not in json.dumps(trace.get("line", {})) and "#" not in json.dumps(
                trace.get("marker", {})
            )
        assert f'id="{chart_id}"' in html
    equity_spec = json.loads(specs[0][1])
    assert [t["name"] for t in equity_spec["data"]] == ["strategy", "benchmark (rebased)"]
    assert json.loads(specs[2][1])["layout"]["xaxis"]["type"] == "category"
    assert html.count("<details>") == 4 and "Table view" in html
    assert "plotly.js/2.35.2/plotly.min.js" in html and "THEMES" in html
    assert "Block-bootstrap Sharpe interval" in html and "Deflated Sharpe ratio" in html
    assert "Walk-forward" not in html
    plain = build_tearsheet(res)
    assert "headline Sharpe is in-sample" in plain and "Deflated Sharpe ratio" not in plain
    assert "Chart library not loaded" in plain and 'src="https://cdnjs' in plain
    offline = build_tearsheet(res, inline_plotly=True)
    assert 'src="https://cdnjs' not in offline and len(offline) > 1_000_000 and "Plotly" in offline
