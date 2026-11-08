"""End-to-end: the synthetic demo runs every stage offline; every CLI command answers --help; the examples
compile."""

from __future__ import annotations

import json
import py_compile
from datetime import date
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from quant.cli.main import app
from quant.config.settings import load_settings
from quant.demo import DEMO_VENDOR, SYNTHETIC_NOTICE, Demo

runner = CliRunner()


def _command_paths() -> list[list[str]]:
    out: list[list[str]] = []

    def walk(cmd, prefix):  # noqa: ANN001
        if hasattr(cmd, "commands"):
            for name, sub in sorted(cmd.commands.items()):
                walk(sub, [*prefix, name])
        else:
            out.append(prefix)

    walk(get_command(app), [])
    return out


@pytest.mark.parametrize("path", _command_paths(), ids=lambda p: " ".join(p))
def test_every_cli_command_answers_help(path):
    r = runner.invoke(app, [*path, "--help"])
    assert r.exit_code == 0, r.output
    assert "Usage" in r.output


def test_cli_command_inventory_is_complete():
    names = {" ".join(p) for p in _command_paths()}
    expected = {
        "demo",
        "version",
        "config",
        "screen",
        "data fetch",
        "data adapters",
        "data filings",
        "universe build",
        "backtest run",
        "analyze regime",
        "fundamentals dcf",
        "factors build",
        "macro dashboard",
        "optimize weights",
        "risk report",
        "paper run",
        "paper live",
    }
    assert expected <= names, expected - names


def test_demo_runs_every_stage_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(paths={"data_dir": str(tmp_path / "data")})
    demo = Demo(settings, tmp_path / "out", seed=7, start=date(2021, 1, 4), end=date(2024, 6, 28))
    result = demo.run()
    failed = [s for s in result.steps if not s.ok]
    assert not failed, "\n".join(f"{s.name}: {s.detail}" for s in failed)
    names = [s.name for s in result.steps]
    assert names == [
        "dataset",
        "validate",
        "universe",
        "screen",
        "momentum",
        "factor_tilt",
        "regime",
        "factors",
        "optimise",
        "risk",
        "macro",
        "paper",
    ]
    assert result.index is not None and result.index.exists()
    index = result.index.read_text()
    assert "SYNTHETIC" in index and "tearsheet_xs_momentum.html" in index and "blotter.html" in index
    for name in (
        "validation",
        "screen",
        "tearsheet_xs_momentum",
        "tearsheet_factor_tilt",
        "tearsheet_regime_switch",
        "factors",
        "weights_hrp",
        "risk",
        "macro_board",
        "blotter",
    ):
        html = (tmp_path / "out" / f"{name}.html").read_text()
        assert "Caveats" in html and "SYNTHETIC" in html, name
    manifest = json.loads((tmp_path / "data" / "demo_manifest.json").read_text())
    assert (
        manifest["vendor"] == DEMO_VENDOR and manifest["notice"] == SYNTHETIC_NOTICE and manifest["seed"] == 7
    )
    assert len(manifest["instruments"]) == 24 and manifest["actions"] > 10
    # the dataset is deterministic: a second build with the same seed changes nothing
    first = demo.svc.parquet.read_series("XNAS_US01", DEMO_VENDOR)
    demo2 = Demo(
        load_settings(paths={"data_dir": str(tmp_path / "data")}),
        tmp_path / "out2",
        seed=7,
        start=date(2021, 1, 4),
        end=date(2024, 6, 28),
        backtests=False,
    )
    demo2.step_dataset()
    second = demo2.svc.parquet.read_series("XNAS_US01", DEMO_VENDOR)
    assert first["close"].tolist() == second["close"].tolist()
    # the planted split is visible in the raw prices and in the validation report
    us01 = first.set_index(first["session_date"].dt.date)["close"]
    assert us01[date(2021, 5, 28)] / us01[date(2021, 6, 1)] > 3.0
    summary = result.summary()
    assert "all steps passed" in summary and "SYNTHETIC" in summary


def test_demo_cli_quick(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("QUANT_PATHS__DATA_DIR", str(tmp_path / "data"))
    r = runner.invoke(
        app,
        ["demo", "--out", str(tmp_path / "rep"), "--seed", "3", "--start", "2022-01-03", "--no-backtests"],
    )
    assert r.exit_code == 0, r.output
    assert (
        "all steps passed" in r.output
        and "SYNTHETIC" in r.output
        and (tmp_path / "rep" / "index.html").exists()
    )


def test_examples_compile():
    for path in sorted(Path("examples").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
