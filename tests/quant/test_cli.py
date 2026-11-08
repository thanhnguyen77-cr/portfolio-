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
