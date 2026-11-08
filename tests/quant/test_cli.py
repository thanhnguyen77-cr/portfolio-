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
    assert "Research use only" in result.output
