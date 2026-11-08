"""The offline end-to-end demo: synthetic dataset -> validate -> universe -> screen -> backtests -> factors
-> optimise -> risk -> macro board -> one paper-trading session -> an index page linking every report.

Every step is wrapped so a failure is recorded and reported, never hidden, and the run continues. Every
report produced here carries the SYNTHETIC DATA notice in its caveats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant.config.settings import QuantSettings
from quant.core.time import now
from quant.data.service import DataService
from quant.demo.dataset import (
    DEMO_END,
    DEMO_SEED,
    DEMO_START,
    DEMO_VENDOR,
    SYNTHETIC_NOTICE,
    DemoAdapter,
    DemoManifest,
    build_demo_dataset,
)
from quant.reports.builder import Section, render_report


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str
    seconds: float
    report: Path | None = None


@dataclass
class DemoResult:
    out_dir: Path
    manifest: DemoManifest
    steps: list[StepResult] = field(default_factory=list)
    index: Path | None = None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def summary(self) -> str:
        lines = [f"demo ({'all steps passed' if self.ok else 'SOME STEPS FAILED'}) -> {self.out_dir}"]
        for s in self.steps:
            mark = "ok  " if s.ok else "FAIL"
            lines.append(f"  [{mark}] {s.name:<12} {s.seconds:5.1f}s  {s.detail}")
        if self.index:
            lines.append(f"  index: {self.index}")
        lines.append(f"  {SYNTHETIC_NOTICE}")
        return "\n".join(lines)


class Demo:
    def __init__(
        self,
        settings: QuantSettings,
        out_dir: Path | str,
        *,
        seed: int = DEMO_SEED,
        start: date = DEMO_START,
        end: date = DEMO_END,
        backtests: bool = True,
    ) -> None:
        settings.data.adapter_priority[:] = [DEMO_VENDOR] + [
            a for a in settings.data.adapter_priority if a != DEMO_VENDOR
        ]
        self.settings = settings
        self.svc = DataService(settings)
        self.svc.adapters[DEMO_VENDOR] = DemoAdapter()
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.seed, self.start, self.end, self.backtests = seed, start, end, backtests
        self.bt_start = max(start + timedelta(days=400), start)
        self.manifest: DemoManifest | None = None
        self.caveats = [SYNTHETIC_NOTICE]

    # ------------------------------------------------------------------ steps
    def _run(self, result: DemoResult, name: str, fn) -> None:  # noqa: ANN001
        t0 = time.monotonic()
        try:
            detail, report = fn()
            result.steps.append(StepResult(name, True, detail, time.monotonic() - t0, report))
        except Exception as exc:  # noqa: BLE001 - the demo reports failures instead of dying
            result.steps.append(
                StepResult(name, False, f"{type(exc).__name__}: {str(exc)[:160]}", time.monotonic() - t0)
            )

    def step_dataset(self) -> tuple[str, Path | None]:
        self.manifest = build_demo_dataset(self.svc, seed=self.seed, start=self.start, end=self.end)
        return (
            f"{len(self.manifest.instruments)} series, {self.manifest.actions} actions, "
            f"{len(self.manifest.macro_series)} macro series (seed {self.seed})",
            self.settings.paths.data_dir / "demo_manifest.json",
        )

    def _equities(self) -> list[str]:
        assert self.manifest is not None
        return self.manifest.universes["demo_all"]

    def step_validate(self) -> tuple[str, Path | None]:
        report = self.svc.validate(self._equities(), self.start, self.end, refresh=False)
        grades = pd.Series({iid: q.grade for iid, q in report.results.items()})
        issues = sum(len(q.issues) for q in report.results.values())
        table = pd.DataFrame(
            [
                {
                    "instrument_id": iid,
                    "grade": q.grade,
                    "issues": len(q.issues),
                    "first": "; ".join(str(i) for i in q.issues[:2]),
                }
                for iid, q in report.results.items()
            ]
        )
        path = self.out / "validation.html"
        render_report(
            "Data validation (synthetic demo)", [Section("Grades", table=table)], self.caveats, path=path
        )
        n_actions = len(self.svc.db.load_corporate_actions(self._equities()))
        return (
            f"grades {grades.value_counts().to_dict()}, {issues} issue(s); {n_actions} corporate actions "
            "on file (the planted 4:1 split is a recorded action, so it is not an anomaly)",
            path,
        )

    def step_universe(self) -> tuple[str, Path | None]:
        uni = self.svc.universe("demo_all")
        members = uni.membership_at(self.end)
        cov = uni.coverage(self.start, self.end, set(members))
        return f"{len(members)} members, quality {uni.quality()}, coverage {cov:.0%}", None

    def step_screen(self) -> tuple[str, Path | None]:
        from quant.universe.screen import ScreenConfig

        cfg = ScreenConfig(min_adv_base=1e5, min_sessions=120)
        table = self.svc.screen(self._equities(), self.start, self.end, config=cfg, refresh=False)
        path = self.out / "screen.html"
        render_report("Screen (synthetic demo)", [Section("Ranking", table=table)], self.caveats, path=path)
        passing = int(table["passes"].sum()) if "passes" in table.columns else len(table)
        return f"{len(table)} ranked, {passing} pass the screen", path

    def _backtest(self, strategy, name: str, base: str, universe: str | None):  # noqa: ANN001,ANN202
        from quant.backtest.analysis import block_bootstrap_sharpe, deflated_sharpe_ratio
        from quant.backtest.engine import BacktestConfig, trial_sharpes, variant_count
        from quant.backtest.runner import run_from_service
        from quant.reports.tearsheet import build_tearsheet

        cfg = BacktestConfig(start=self.bt_start, end=self.end, initial_cash=1_000_000.0, base_currency=base)
        res = run_from_service(self.svc, strategy, cfg, refresh=False, universe=universe)
        res.caveats.insert(0, SYNTHETIC_NOTICE)
        robustness: dict[str, Any] = {}
        rets = res.equity_series.pct_change().dropna()
        if len(rets) > 60:
            robustness["bootstrap"] = block_bootstrap_sharpe(rets, n_boot=200, seed=self.seed)
            robustness["deflated_sharpe"] = deflated_sharpe_ratio(
                rets, variant_count(strategy.name), trial_sharpes=trial_sharpes(strategy.name)
            )
        path = self.out / f"tearsheet_{name}.html"
        build_tearsheet(res, robustness=robustness, path=path)
        res.to_dir(self.out / f"run_{name}")
        m = res.metrics
        nan = float("nan")
        detail = (
            f"CAGR {m.get('cagr', nan):+.1%}, Sharpe {m.get('sharpe', nan):.2f}, "
            f"max DD {m.get('max_drawdown', nan):.1%}, {len(res.fills)} fills"
        )
        return detail, path

    def step_backtest_momentum(self) -> tuple[str, Path | None]:
        from quant.strategies import CrossSectionalMomentum

        return self._backtest(
            CrossSectionalMomentum(self._equities(), top_n=6), "xs_momentum", "GBP", "demo_all"
        )

    def step_backtest_factor_tilt(self) -> tuple[str, Path | None]:
        from quant.strategies import FactorTilt

        return self._backtest(
            FactorTilt(self._equities(), top_fraction=0.3), "factor_tilt", "USD", "demo_all"
        )

    def step_backtest_regime(self) -> tuple[str, Path | None]:
        from quant.strategies import MacroRegimeSwitch

        assert self.manifest is not None
        us = self.manifest.universes["demo_us"][:4]
        ids = us + ["XNYS_SPX", "XNYS_VIX"]
        strat = MacroRegimeSwitch(
            ids, benchmark="XNYS_SPX", vol_index="XNYS_VIX", risk_on={i: 0.9 / len(us) for i in us}
        )
        return self._backtest(strat, "regime_switch", "USD", None)

    def step_factors(self) -> tuple[str, Path | None]:
        fr = self.svc.build_factor_returns(
            self._equities(),
            self.bt_start,
            self.end,
            factors=("momentum", "low_vol"),
            n_quantiles=3,
            refresh=False,
        )
        rets = fr.returns if hasattr(fr, "returns") else fr
        path = self.out / "factors.html"
        table = rets.describe().T.reset_index().rename(columns={"index": "factor"})
        render_report(
            "Factor returns (synthetic demo)",
            [Section("Daily long-short returns", table=table)],
            self.caveats + list(getattr(fr, "notes", [])),
            path=path,
        )
        return f"{list(rets.columns)} over {len(rets)} days", path

    def step_optimise(self) -> tuple[str, Path | None]:
        from quant.portfolio import hrp, ledoit_wolf_cov, returns_from_prices, risk_contributions

        panel = self.svc.close_panel(
            self._equities(), self.bt_start, self.end, to_currency="GBP", refresh=False
        ).dropna(how="any")
        rets = returns_from_prices(panel)
        est = ledoit_wolf_cov(rets)
        res = hrp(est.cov)
        table = pd.DataFrame(
            {
                "weight": res.weights,
                "risk_contribution": risk_contributions(res.weights, est.cov),
                "vol": est.vol(),
            }
        )
        path = self.out / "weights_hrp.html"
        render_report(
            "HRP weights (synthetic demo)",
            [
                Section(
                    "Weights",
                    table=table.rename_axis("instrument").reset_index(),
                    kv={"ex-ante vol": f"{res.volatility:.2%}"},
                )
            ],
            self.caveats + est.notes,
            path=path,
        )
        self._hrp_weights = res.weights
        return f"HRP over {len(res.weights)} names, ex-ante vol {res.volatility:.1%}", path

    def step_risk(self) -> tuple[str, Path | None]:
        from quant.portfolio import ledoit_wolf_cov, returns_from_prices
        from quant.risk import build_risk_report

        panel = self.svc.close_panel(
            self._equities(), self.start, self.end, to_currency="GBP", refresh=False
        ).dropna(how="any")
        rets = returns_from_prices(panel)
        w = getattr(self, "_hrp_weights", pd.Series(1.0 / panel.shape[1], index=panel.columns))
        path = self.out / "risk.html"
        build_risk_report(
            rets,
            w,
            prices=panel,
            cov=ledoit_wolf_cov(rets).cov,
            extra_caveats=self.caveats,
            path=path,
            title="Risk report (synthetic demo)",
        )
        return f"VaR family, stress windows and drawdowns on {len(rets)} days", path

    def step_macro(self) -> tuple[str, Path | None]:
        board = self.svc.macro_dashboard("us", end=self.end, refresh=False)
        path = self.out / "macro_board.html"
        render_report(
            "Macro board (synthetic demo)",
            [
                Section(
                    "Indicators",
                    table=board["table"],
                    kv={"quadrant": board["quadrant"], "signals": str(board["risk_signals"])},
                )
            ],
            self.caveats + list(board["notes"]) + [f"missing inputs: {board['missing'] or 'none'}"],
            path=path,
        )
        return f"quadrant '{board['quadrant']}', {len(board['table'])} indicators", path

    def step_paper(self) -> tuple[str, Path | None]:
        from quant.backtest.costs import CostModel
        from quant.core.calendars import get_calendar
        from quant.paper import Ledger, PaperRunner, PreTradeRules, SimBroker, build_blotter
        from quant.strategies import FixedWeights

        assert self.manifest is not None
        us = self.manifest.universes["demo_us"][:4]
        cal = get_calendar("XNAS")
        sessions = cal.sessions(self.end - timedelta(days=10), self.end)
        first, last = sessions[-3].date(), sessions[-1].date()
        ledger = Ledger(self.svc.db)
        account = f"demo_{self.seed}"
        if ledger.exists(account):
            ledger.delete_account(account)
        state = ledger.create_account(
            account, broker="sim", base_currency="USD", initial_cash=1_000_000.0, as_of=sessions[-4].date()
        )
        broker = SimBroker(state, {i: self.svc.resolve(i) for i in us}, cost_model=CostModel(), ledger=ledger)
        runner = PaperRunner(
            self.svc,
            broker,
            FixedWeights(us, {i: 0.08 for i in us}, rebalance="never"),
            PreTradeRules(),
            refresh=False,
        )
        r1 = runner.run_session(first)
        r2 = runner.run_session(last)
        path = self.out / "blotter.html"
        build_blotter(state, broker, reconciliation=r2.reconciliation, path=path, extra_caveats=self.caveats)
        ok = r2.reconciliation is not None and r2.reconciliation.ok
        return (
            f"{len(r1.submitted)} orders submitted on {first}, {len(r2.fills)} filled by {last}, "
            f"reconciled={ok}",
            path,
        )

    def step_index(self, result: DemoResult) -> Path:
        rows = [
            {
                "step": s.name,
                "status": "ok" if s.ok else "FAILED",
                "seconds": round(s.seconds, 1),
                "detail": s.detail,
                "report": s.report.name if s.report else "",
            }
            for s in result.steps
        ]
        links = "".join(
            f'<li><a href="{s.report.name}">{s.report.name}</a></li>'
            for s in result.steps
            if s.report and s.report.suffix == ".html"
        )
        path = self.out / "index.html"
        render_report(
            "quant demo (synthetic data)",
            [Section("Steps", table=pd.DataFrame(rows)), Section("Reports", html=f"<ul>{links}</ul>")],
            self.caveats
            + ["Run with `quant demo`; regenerate with another --seed to see how much the numbers move."],
            subtitle=f"generated {now():%Y-%m-%d %H:%M} UTC",
            path=path,
        )
        return path

    # ------------------------------------------------------------------ run
    def run(self) -> DemoResult:
        result = DemoResult(self.out, DemoManifest(self.seed, self.start, self.end))
        self._run(result, "dataset", self.step_dataset)
        if self.manifest is None:
            result.index = self.step_index(result)
            return result
        result.manifest = self.manifest
        self._run(result, "validate", self.step_validate)
        self._run(result, "universe", self.step_universe)
        self._run(result, "screen", self.step_screen)
        if self.backtests:
            self._run(result, "momentum", self.step_backtest_momentum)
            self._run(result, "factor_tilt", self.step_backtest_factor_tilt)
            self._run(result, "regime", self.step_backtest_regime)
        self._run(result, "factors", self.step_factors)
        self._run(result, "optimise", self.step_optimise)
        self._run(result, "risk", self.step_risk)
        self._run(result, "macro", self.step_macro)
        self._run(result, "paper", self.step_paper)
        result.index = self.step_index(result)
        return result
