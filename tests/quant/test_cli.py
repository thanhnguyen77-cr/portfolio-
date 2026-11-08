from __future__ import annotations

from typer.testing import CliRunner

from quant import __version__
from quant.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output


def test_config_shows_effective_settings_with_secrets_masked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_FRED_API_KEY", "topsecret")
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.output
    assert '"live_enabled": false' in result.output
    assert "topsecret" not in result.output
    assert "**********" in result.output


def test_no_args_shows_help():
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Research use only" in " ".join(result.output.split())  # help text is line-wrapped


def test_data_instruments_lists_catalogue(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["data", "instruments", "--market", "XLON"])
    assert result.exit_code == 0, result.output
    assert "XLON_VOD" in result.output and "XNAS_AAPL" not in result.output


def test_data_fetch_reports_blocked_vendor_and_records_lineage(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["data", "fetch", "AAPL", "--start", "2024-07-01", "--end", "2024-07-12"])
    assert result.exit_code == 1, result.output
    assert "XNAS_AAPL" in result.output and "blocked" in result.output
    cached = runner.invoke(app, ["data", "cached"])
    assert cached.exit_code == 0 and "XNAS_AAPL" in cached.output
    lineage = runner.invoke(app, ["data", "lineage"])
    assert lineage.exit_code == 0 and "fetch" in lineage.output


def test_data_validate_on_cached_bars(monkeypatch, tmp_path):
    from datetime import date

    from conftest import make_bars

    from quant.core.catalog import load_default_catalog
    from quant.data.store import ParquetStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    cat = load_default_catalog()
    bars = make_bars(cat["XNAS_AAPL"], date(2024, 1, 2), date(2024, 12, 31), vendor="yfinance")
    ParquetStore(tmp_path / "data" / "parquet").write_bars(bars)
    args = ["data", "validate", "AAPL", "--start", "2024-01-01", "--end", "2024-12-31", "--issues"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "XNAS_AAPL" in result.output and "data quality" in result.output


def test_data_macro_blocked_vendor_exits_1(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["data", "macro", "DGS10", "--start", "2024-07-01", "--end", "2024-07-05"])
    assert result.exit_code == 1 and "no observations" in result.output


def test_universe_import_show_and_screen_offline(monkeypatch, tmp_path):
    from datetime import date

    from conftest import make_bars

    from quant.core.catalog import load_default_catalog
    from quant.data.store import ParquetStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    csv = tmp_path / "members.csv"
    csv.write_text("instrument_id,effective_from,effective_to\nXNAS_AAPL,2020-01-01,\nXNYS_JPM,2020-01-01,\n")
    res = runner.invoke(app, ["universe", "import", "demo", str(csv), "--complete"])
    assert res.exit_code == 0, res.output
    assert "pit_complete" in res.output
    show = runner.invoke(app, ["universe", "show", "demo", "--as-of", "2024-06-01"])
    assert show.exit_code == 0 and "XNAS_AAPL" in show.output and "2 members" in show.output
    cat = load_default_catalog()
    store = ParquetStore(tmp_path / "data" / "parquet")
    for iid, seed in (("XNAS_AAPL", 1), ("XNYS_JPM", 2)):
        store.write_bars(
            make_bars(cat[iid], date(2024, 1, 2), date(2024, 12, 31), seed=seed, vendor="yfinance")
        )
    out = tmp_path / "screen.html"
    scr = runner.invoke(
        app,
        [
            "screen",
            "demo",
            "--start",
            "2024-01-02",
            "--end",
            "2024-12-31",
            "--no-refresh",
            "--html",
            str(out),
        ],
    )
    assert scr.exit_code == 0, scr.output
    assert "screen (" in scr.output and out.exists() and "Caveats" in out.read_text()


def test_universe_build_without_network_fails_cleanly(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    res = runner.invoke(app, ["universe", "build", "ftse100"])
    assert res.exit_code == 1 and "could not build ftse100" in res.output


def test_backtest_run_cli_with_robustness(monkeypatch, tmp_path):
    from datetime import date

    from conftest import make_bars

    from quant.core.catalog import load_default_catalog
    from quant.data.store import ParquetStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    cat = load_default_catalog()
    store = ParquetStore(tmp_path / "data" / "parquet")
    for iid, seed in (("XNAS_AAPL", 1), ("XNYS_JPM", 2)):
        store.write_bars(
            make_bars(cat[iid], date(2021, 1, 4), date(2024, 6, 28), seed=seed, vendor="yfinance")
        )
    out = tmp_path / "tearsheet.html"
    res = runner.invoke(
        app,
        [
            "backtest",
            "run",
            "trend_following",
            "--instruments",
            "AAPL,JPM",
            "--start",
            "2023-01-03",
            "--end",
            "2024-06-28",
            "--param",
            "fast=5",
            "--param",
            "slow=20",
            "--no-refresh",
            "--bootstrap",
            "--sensitivity",
            "fast=5,10;slow=20,40",
            "--html",
            str(out),
            "--out",
            str(tmp_path / "run"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert (
        "trend_following" in res.output
        and "deflated Sharpe" in res.output
        and "parameter sensitivity" in res.output
    )
    assert out.exists() and (tmp_path / "run" / "result.json").exists()
    bad = runner.invoke(app, ["backtest", "run", "nope", "--instruments", "AAPL", "--no-refresh"])
    assert bad.exit_code != 0


def test_analyze_commands_on_cached_bars(monkeypatch, tmp_path):
    from datetime import date

    from conftest import make_bars

    from quant.core.catalog import load_default_catalog
    from quant.data.store import ParquetStore

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    cat = load_default_catalog()
    store = ParquetStore(tmp_path / "data" / "parquet")
    for iid, seed in (("XNAS_AAPL", 1), ("XNYS_JPM", 2)):
        store.write_bars(
            make_bars(cat[iid], date(2019, 1, 2), date(2024, 6, 28), seed=seed, vendor="yfinance")
        )
    common = ["--start", "2019-01-02", "--end", "2024-06-28", "--no-refresh"]
    checks = {
        ("indicators", "AAPL"): "rsi14",
        ("levels", "AAPL"): "level",
        ("regime", "AAPL", "--markov"): "risk_",
        ("stationarity", "AAPL"): "verdict",
        ("cointegration", "AAPL", "JPM"): "hedge_ratio",
        ("garch", "AAPL", "--dist", "normal"): "persistence",
        ("pca", "AAPL", "JPM"): "pc1",
    }
    for args, needle in checks.items():
        res = runner.invoke(app, ["analyze", *args, *common])
        assert res.exit_code == 0, (args, res.output)
        assert needle in res.output, (args, res.output)
    missing = runner.invoke(app, ["analyze", "indicators", "XOM", *common])
    assert missing.exit_code == 1 and "no bars" in missing.output


def test_fundamentals_commands_on_cached_data(monkeypatch, tmp_path):
    from datetime import date

    from conftest import make_bars, yf_statement_frames

    from quant.core.catalog import load_default_catalog
    from quant.data.store import DuckDBStore, ParquetStore
    from quant.fundamental import from_yfinance

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    cat = load_default_catalog()
    db = DuckDBStore(tmp_path / "data" / "quant.duckdb")
    db.migrate()
    db.upsert_fundamentals(from_yfinance(cat["XNAS_AAPL"], yf_statement_frames()))
    db.upsert_fundamentals(from_yfinance(cat["XNYS_JPM"], yf_statement_frames(scale=2.0)))
    store = ParquetStore(tmp_path / "data" / "parquet")
    store.write_bars(
        make_bars(cat["XNAS_AAPL"], date(2024, 1, 2), date(2024, 8, 30), seed=1, vendor="yfinance")
    )
    ed = yf_statement_frames()["earnings_dates"]
    import pandas as pd

    from quant.core.time import now

    db.upsert_earnings(
        pd.DataFrame(
            {
                "instrument_id": "XNAS_AAPL",
                "report_date": ed.index.tz_localize(None).normalize(),
                "eps_estimate": ed["EPS Estimate"].to_numpy(dtype=float),
                "eps_actual": pd.to_numeric(ed["Reported EPS"]).to_numpy(dtype=float),
                "currency": "USD",
                "vendor": "yfinance",
                "recorded_at": pd.Timestamp(now()),
            }
        )
    )
    r = runner.invoke(app, ["fundamentals", "ratios", "AAPL", "--as-of", "2024-08-30"])
    assert r.exit_code == 0, r.output
    assert "roic" in r.output and "pe" in r.output
    d = runner.invoke(app, ["fundamentals", "dcf", "AAPL", "--rate", "0.09", "--as-of", "2024-08-30"])
    assert d.exit_code == 0, d.output
    assert "per_share" in d.output and "terminal growth" in d.output
    s = runner.invoke(app, ["fundamentals", "scores", "AAPL", "JPM", "--as-of", "2024-08-30"])
    assert s.exit_code == 0, s.output
    assert "composite" in s.output
    e = runner.invoke(app, ["fundamentals", "earnings", "AAPL"])
    assert e.exit_code == 0, e.output
    assert "beat_rate" in e.output
    missing = runner.invoke(app, ["fundamentals", "ratios", "VOD.L"])
    assert missing.exit_code == 1 and "no cached statements" in missing.output
    blocked = runner.invoke(app, ["fundamentals", "fetch", "XOM"])
    assert blocked.exit_code == 1 and "blocked" in blocked.output
