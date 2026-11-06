"""Tests for the Trader skill scripts. Run: python3 -m unittest discover -s tests -v"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins" / "trader" / "skills" / "trader" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import traderlib as tl  # noqa: E402
import backtest as bt  # noqa: E402
import kelly  # noqa: E402
import risk_check  # noqa: E402
import screen  # noqa: E402


def synth_bars(n=900, drift=0.001, vol=0.015, seed=7, start=100.0):
    r = random.Random(seed)
    px, d, out = start, date(2022, 1, 3), []
    while len(out) < n:
        if d.weekday() < 5:
            o = px * (1 + r.gauss(0, vol / 3))
            c = px * math.exp(r.gauss(drift, vol))
            h = max(o, c) * (1 + abs(r.gauss(0, vol / 2)))
            lo = min(o, c) * (1 - abs(r.gauss(0, vol / 2)))
            out.append({"time": d.isoformat(), "open": o, "high": h, "low": lo, "close": c,
                        "volume": 1e6 * (1 + abs(r.gauss(0, 0.4)))})
            px = c
        d += timedelta(days=1)
    return out


def make_profile(**over):
    p = tl.merge_defaults({"budget": 10_000, "cash_reserve": 2_000, "max_position_value": 2_500})
    p.update(over)
    return p


def run(script, *args, env=None, stdin=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True,
                          env=e, input=stdin)


class IndicatorTests(unittest.TestCase):
    def test_sma_ema(self):
        v = [1, 2, 3, 4, 5]
        self.assertEqual(tl.sma(v, 3), [None, None, 2.0, 3.0, 4.0])
        e = tl.ema(v, 3)
        self.assertEqual(e[2], 2.0)
        self.assertAlmostEqual(e[3], 3.0)

    def test_rsi_extremes(self):
        up = [float(i) for i in range(30)]
        self.assertEqual(tl.rsi(up, 14)[-1], 100.0)
        down = [float(30 - i) for i in range(30)]
        self.assertEqual(tl.rsi(down, 14)[-1], 0.0)
        self.assertIsNone(tl.rsi(up, 14)[13])

    def test_rsi_matches_reference(self):
        # Wilder RSI reference values (classic 14-period example)
        closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
                  45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]
        r = tl.rsi(closes, 14)
        self.assertAlmostEqual(r[14], 70.46, delta=0.1)
        self.assertAlmostEqual(r[19], 57.92, delta=0.6)

    def test_atr_constant_range(self):
        h, l, c = [11.0] * 30, [9.0] * 30, [10.0] * 30
        self.assertAlmostEqual(tl.atr(h, l, c, 14)[-1], 2.0)

    def test_wilson_lower_bound(self):
        self.assertLess(tl.wilson_lower_bound(0.6, 10), tl.wilson_lower_bound(0.6, 1000))
        self.assertLess(tl.wilson_lower_bound(0.6, 1000), 0.6)


class ParseTests(unittest.TestCase):
    def test_shapes(self):
        rows = [{"t": 1700000000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                {"t": 1700086400, "o": 1.5, "h": 2.5, "l": 1, "c": 2, "v": 12}]
        a = tl.parse_bars(json.dumps({"bars": {"AAPL": rows}}), symbol="AAPL")
        b = tl.parse_bars(json.dumps({"t": [1700000000, 1700086400], "o": [1, 1.5], "h": [2, 2.5],
                                      "l": [0.5, 1], "c": [1.5, 2], "v": [10, 12], "s": "ok"}))
        c = tl.parse_bars(json.dumps([[1700000000, 1, 2, 0.5, 1.5, 10], [1700086400, 1.5, 2.5, 1, 2, 12]]))
        d = tl.parse_bars("Date,Open,High,Low,Close,Volume\n2023-11-15,1.5,2.5,1,2,12\n2023-11-14,1,2,0.5,1.5,10\n")
        for bars in (a, b, c, d):
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars[-1]["close"], 2.0)
            self.assertLess(bars[0]["time"], bars[1]["time"])

    def test_millisecond_timestamps(self):
        bars = tl.parse_bars(json.dumps([{"time": 1700000000000, "close": 5}]))
        self.assertTrue(bars[0]["time"].startswith("2023-11-14"))
        self.assertEqual(bars[0]["open"], 5)


class BacktestFillTests(unittest.TestCase):
    def _flat(self, n):
        # 260 quiet bars in a gentle uptrend so SMAs exist and ATR ~ 1
        out = []
        for i in range(n):
            c = 100 + i * 0.05
            out.append({"time": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
                        "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1e6})
        return out

    def test_stop_wins_when_both_hit_same_bar(self):
        bars = self._flat(260)
        p = bt.Params(setup="breakout", stop_atr=2, rr=2, max_hold=10, slippage_bps=0, fee_bps=0,
                      capital=10_000, position_value=10_000)
        f = tl.indicator_frame(bars)
        calls = {"n": 0}

        def one_signal(i):
            if i == 255 and calls["n"] == 0:
                calls["n"] += 1
                return True
            return False
        orig = bt.make_signal
        bt.make_signal = lambda *_a, **_k: one_signal
        try:
            entry_open = bars[256]["open"]
            atr = f["atr14"][255]
            bars[256]["high"] = entry_open + 10 * atr   # hits target ...
            bars[256]["low"] = entry_open - 10 * atr    # ... and stop on the same bar
            f = tl.indicator_frame(bars)
            sim = bt.simulate("X", bars, p, f)
        finally:
            bt.make_signal = orig
        self.assertEqual(len(sim["trades"]), 1)
        t = sim["trades"][0]
        self.assertEqual(t.exit_reason, "stop")
        self.assertAlmostEqual(t.r_multiple, -1.0, places=6)

    def test_gap_through_stop_fills_at_open(self):
        bars = self._flat(260)
        p = bt.Params(setup="breakout", stop_atr=2, rr=3, max_hold=10, slippage_bps=0, fee_bps=0)
        fired = {"done": False}

        def sig(i):
            if i == 250 and not fired["done"]:
                fired["done"] = True
                return True
            return False
        orig = bt.make_signal
        bt.make_signal = lambda *_a, **_k: sig
        try:
            gap = bars[251]["open"] - 20
            bars[253].update(open=gap, high=gap + 0.1, low=gap - 0.1, close=gap)
            sim = bt.simulate("X", bars, p)
        finally:
            bt.make_signal = orig
        t = sim["trades"][0]
        self.assertEqual(t.exit_reason, "gap_stop")
        self.assertAlmostEqual(t.exit_price, gap)
        self.assertLess(t.r_multiple, -1.0)

    def test_costs_reduce_pnl(self):
        bars = synth_bars(800, seed=3)
        free = bt.simulate("X", bars, bt.Params(setup="mean_reversion", slippage_bps=0, fee_bps=0))
        costly = bt.simulate("X", bars, bt.Params(setup="mean_reversion", slippage_bps=20, fee_bps=5,
                                                  commission_per_order=1))
        self.assertGreater(len(free["trades"]), 0)
        self.assertLess(sum(t.pnl for t in costly["trades"]), sum(t.pnl for t in free["trades"]))

    def test_no_lookahead_in_signals(self):
        bars = synth_bars(700, seed=11)
        full = tl.indicator_frame(bars)
        for setup in bt.SETUPS:
            sig_full = bt.make_signal(setup, full, bt.Params(setup=setup))
            cut = 500
            part = tl.indicator_frame(bars[:cut + 1])
            sig_part = bt.make_signal(setup, part, bt.Params(setup=setup))
            for i in range(250, cut + 1):
                self.assertEqual(sig_full(i), sig_part(i), f"{setup} signal at {i} depends on future bars")

    def test_cli_plan_and_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ABC.csv"
            tl.write_bars_csv(synth_bars(1000, seed=5), path)
            plan = Path(d) / "plan.json"
            plan.write_text(json.dumps({"symbol": "ABC", "setup": "pullback", "entry": 100, "stop": 96,
                                        "target": 110, "atr14": 2.0}))
            out = Path(d) / "bt.json"
            r = run("backtest.py", "--bars", str(path), "--plan", str(plan), "--out", str(out),
                    env={"TRADER_WORKSPACE": d})
            self.assertEqual(r.returncode, 0, r.stderr)
            res = json.loads(out.read_text())
            self.assertEqual(res["params"]["stop_atr"], 2.0)
            self.assertEqual(res["params"]["rr"], 2.5)
            self.assertIn(res["verdict"], {"PASS", "INSUFFICIENT_TRADES", "FRAGILE", "MARGINAL", "FAIL_OOS", "FAIL"})
            self.assertEqual(len(res["sensitivity"]["grid"]), 9)


class KellyTests(unittest.TestCase):
    def test_formula(self):
        self.assertAlmostEqual(kelly.kelly_fraction(0.5, 2.0), 0.25)
        self.assertLess(kelly.kelly_fraction(0.3, 1.5), 0)

    def test_no_edge_no_trade(self):
        res = kelly.size(p_hat=0.35, payoff=1.5, n_trades=100, entry=50, stop=48, prof=make_profile())
        self.assertEqual(res["decision"], "NO_TRADE")
        self.assertEqual(res["binding_constraint"], "kelly")

    def test_risk_cap_binds(self):
        prof = make_profile()
        res = kelly.size(p_hat=0.6, payoff=2.5, n_trades=500, entry=50, stop=48, prof=prof)
        self.assertEqual(res["decision"], "SIZE")
        self.assertLessEqual(res["dollars_at_risk"], 100.0 + 1e-6)  # 1% of 10k
        self.assertLessEqual(res["position_value"], 2_500)

    def test_cash_reserve_binds(self):
        acct = {"equity": 10_000, "cash": 2_300, "positions": [{"symbol": "A", "market_value": 7_700}]}
        res = kelly.size(p_hat=0.6, payoff=2.5, n_trades=500, entry=10, stop=9.9, prof=make_profile(),
                         account=acct)
        self.assertLessEqual(res["qty"] * 10, 300 + 1e-6)

    def test_regime_off(self):
        res = kelly.size(p_hat=0.6, payoff=2.5, n_trades=500, entry=50, stop=48, prof=make_profile(),
                         regime="risk_off")
        self.assertEqual(res["decision"], "NO_TRADE")

    def test_shrinkage_reduces_kelly(self):
        small = kelly.size(p_hat=0.55, payoff=2, n_trades=20, entry=50, stop=48, prof=make_profile())
        big = kelly.size(p_hat=0.55, payoff=2, n_trades=2000, entry=50, stop=48, prof=make_profile())
        self.assertLess(small["kelly_full_conservative"], big["kelly_full_conservative"])


class RiskTests(unittest.TestCase):
    order = {"symbol": "ABC", "side": "buy", "qty": 10, "order_type": "limit", "limit_price": 100,
             "stop_loss": 95, "take_profit": 112}
    acct = {"equity": 10_000, "last_equity": 10_000, "cash": 9_000,
            "positions": [{"symbol": "XYZ", "qty": 5, "market_value": 1_000}], "open_orders": []}
    bt_pass = {"verdict": "PASS", "symbols": ["ABC"], "all": {"trades": 40}, "params": {"max_hold": 20}}
    sizing = {"decision": "SIZE", "qty": 12}
    news = {"severity": "green", "days_to_earnings": 40}

    def check(self, **over):
        o = {**self.order, **over.pop("order", {})}
        args = dict(account=self.acct, backtest=self.bt_pass, sizing=self.sizing, news=self.news,
                    regime="risk_on")
        args.update(over)
        return risk_check.check_entry(o, make_profile(), args["account"], args["backtest"], args["sizing"],
                                      args["news"], args["regime"])

    def test_good_order_passes(self):
        ck = self.check()
        self.assertEqual(ck.failed, [], ck.items)

    def test_rejections(self):
        cases = {
            "protective_stop": dict(order={"stop_loss": None}),
            "reward_risk": dict(order={"take_profit": 103}),
            "max_position_value": dict(order={"qty": 30, "stop_loss": 99.5, "take_profit": 102}),
            "max_risk_per_trade": dict(order={"qty": 24, "stop_loss": 90, "take_profit": 130}),
            "order_type": dict(order={"order_type": "market"}),
            "backtest": dict(backtest={**self.bt_pass, "verdict": "FAIL_OOS"}),
            "regime": dict(regime="risk_off"),
            "news": dict(news={"severity": "red", "flags": ["SEC probe"]}),
            "earnings": dict(news={"severity": "green", "days_to_earnings": 1}),
            "no_duplicate": dict(order={"symbol": "XYZ"}, backtest={**self.bt_pass, "symbols": ["XYZ"]}),
            "daily_loss_limit": dict(account={**self.acct, "equity": 9_600, "last_equity": 10_000}),
            "account_state": dict(account=None),
            "side": dict(order={"side": "sell"}),
        }
        for name, over in cases.items():
            with self.subTest(name):
                failed = [c["name"] for c in self.check(**over).failed]
                self.assertIn(name, failed)

    def test_cash_reserve(self):
        acct = {**self.acct, "positions": [{"symbol": "XYZ", "market_value": 6_500}], "cash": 2_800}
        failed = [c["name"] for c in self.check(account=acct).failed]
        self.assertIn("cash_reserve", failed)

    def test_stop_limit_uses_trigger_for_rr_and_limit_for_risk(self):
        o = {"order_type": "stop_limit", "stop_price": 100, "limit_price": 100.6, "stop_loss": 95, "take_profit": 110}
        ck = self.check(order=o)
        self.assertEqual(ck.failed, [], ck.items)  # 2.0 R from the trigger passes
        risk = next(c for c in ck.items if c["name"] == "max_risk_per_trade")
        self.assertIn("$56.00", risk["detail"])      # 10 x (100.6 - 95): worst fill
        bad = self.check(order={**o, "limit_price": 99})
        self.assertIn("stop_limit_prices", [c["name"] for c in bad.failed])

    def test_exit_checks(self):
        acct = {**self.acct, "positions": [{"symbol": "ABC", "qty": 10, "market_value": 1_000}]}
        ok = risk_check.check_exit({"symbol": "ABC", "side": "sell", "qty": 10, "order_type": "limit"},
                                   make_profile(), acct)
        self.assertEqual(ok.failed, [])
        bad = risk_check.check_exit({"symbol": "ABC", "side": "sell", "qty": 11, "order_type": "limit"},
                                    make_profile(), acct)
        self.assertIn("qty", [c["name"] for c in bad.failed])

    def test_live_mode_needs_both_switches(self):
        prof = make_profile(mode="live", live_trading_enabled=False)
        self.assertTrue(tl.validate_profile(prof))
        ck = risk_check.check_entry(self.order, make_profile(mode="live", live_trading_enabled=True), self.acct,
                                    self.bt_pass, self.sizing, self.news, "risk_on")
        self.assertIn("mode", [c["name"] for c in ck.failed])  # no per-order user confirmation


class GuardAndCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Path(self.tmp.name) / "trader-workspace"
        self.env = {"TRADER_WORKSPACE": str(self.ws)}
        r = run("trader_profile.py", "init", "--budget", "10000", "--cash-reserve", "2000",
                "--max-position", "2500", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def guard(self, tool, inp):
        payload = json.dumps({"tool_name": tool, "tool_input": inp, "cwd": self.tmp.name})
        return run("order_guard.py", stdin=payload, env=self.env)

    def approve(self):
        d = Path(self.tmp.name)
        (d / "o.json").write_text(json.dumps(RiskTests.order))
        (d / "a.json").write_text(json.dumps(RiskTests.acct))
        (d / "b.json").write_text(json.dumps(RiskTests.bt_pass))
        (d / "s.json").write_text(json.dumps(RiskTests.sizing))
        (d / "n.json").write_text(json.dumps(RiskTests.news))
        return run("risk_check.py", "--order", str(d / "o.json"), "--account", str(d / "a.json"),
                   "--backtest", str(d / "b.json"), "--sizing", str(d / "s.json"), "--news", str(d / "n.json"),
                   "--regime", "risk_on", "--approve", "--out", str(d / "r.json"), env=self.env)

    def test_guard_lifecycle(self):
        order = {"symbol": "ABC", "side": "buy", "qty": 10, "type": "limit", "limit_price": 100,
                 "order_class": "bracket", "stop_loss": {"stop_price": 95}, "take_profit": {"limit_price": 112}}
        self.assertEqual(self.guard("mcp__alpaca__get_account_info", {}).returncode, 0)
        self.assertEqual(self.guard("mcp__alpaca__cancel_order_by_id", {"order_id": "1"}).returncode, 0)
        self.assertEqual(self.guard("mcp__alpaca__place_stock_order", order).returncode, 2)
        r = self.approve()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        no_stop = {k: v for k, v in order.items() if k not in ("order_class", "stop_loss")}
        self.assertEqual(self.guard("mcp__alpaca__place_stock_order", no_stop).returncode, 2)
        worse = {**order, "limit_price": 103}
        self.assertEqual(self.guard("mcp__alpaca__place_stock_order", worse).returncode, 2)
        self.assertEqual(self.guard("mcp__alpaca__place_stock_order", order).returncode, 0)
        self.assertEqual(self.guard("mcp__alpaca__place_stock_order", order).returncode, 2)  # single use
        self.assertEqual(self.guard("mcp__alpaca__place_option_order", order).returncode, 2)

    def test_guard_expired_ticket(self):
        self.assertEqual(self.approve().returncode, 0)
        for f in (self.ws / "risk" / "approvals").glob("*.json"):
            t = json.loads(f.read_text())
            t["expires_at"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            f.write_text(json.dumps(t))
        order = {"symbol": "ABC", "side": "buy", "qty": 10, "limit_price": 100, "order_class": "bracket"}
        r = self.guard("mcp__alpaca__place_stock_order", order)
        self.assertEqual(r.returncode, 2)
        self.assertIn("expired", r.stderr)

    def test_profile_cannot_enable_live(self):
        r = run("trader_profile.py", "set", "mode=live", env=self.env)
        self.assertNotEqual(r.returncode, 0)
        r = run("trader_profile.py", "set", "max_open_positions=3", env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_screen_and_save_bars(self):
        up = synth_bars(400, drift=0.002, vol=0.01, seed=1)
        down = synth_bars(400, drift=-0.002, vol=0.01, seed=2)
        payload = json.dumps({"bars": [{"t": b["time"], "o": b["open"], "h": b["high"], "l": b["low"],
                                        "c": b["close"], "v": b["volume"]} for b in up]})
        r = run("save_bars.py", "--symbol", "NASDAQ:UPPP", stdin=payload, env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        tl.write_bars_csv(down, self.ws / "data" / "DOWN.csv")
        out = self.ws / "screens" / "s.json"
        r = run("screen.py", "--bars-dir", str(self.ws / "data"), "--out", str(out),
                "--md", str(self.ws / "screens" / "s.md"), env=self.env)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(out.read_text())
        by = {x["symbol"]: x for x in res["results"]}
        self.assertIn("UPPP", by)
        self.assertFalse(by["DOWN"]["passed"])
        self.assertIn("close_above_sma50", by["DOWN"]["failed"])

    def test_journal_stats(self):
        run("journal.py", "open", "--symbol", "ABC", "--qty", "10", "--entry", "100", "--stop", "95",
            "--order-id", "o1", env=self.env)
        run("journal.py", "open", "--symbol", "DEF", "--qty", "10", "--entry", "50", "--stop", "48",
            "--order-id", "o2", env=self.env)
        run("journal.py", "close", "--order-id", "o1", "--exit", "110", "--reason", "target", env=self.env)
        run("journal.py", "close", "--order-id", "o2", "--exit", "48", "--reason", "stop", env=self.env)
        run("journal.py", "open", "--symbol", "GHI", "--qty", "5", "--entry", "20", "--stop", "19",
            "--order-id", "o3", env=self.env)
        run("journal.py", "cancel", "--order-id", "o3", env=self.env)
        r = run("journal.py", "stats", env=self.env)
        st = json.loads(r.stdout)
        self.assertEqual(st["closed_trades"], 2)
        self.assertEqual(st["cancelled_unfilled"], 1)
        self.assertAlmostEqual(st["win_rate"], 0.5)
        self.assertAlmostEqual(st["expectancy_r"], 0.5)  # (+2R + -1R) / 2


if __name__ == "__main__":
    unittest.main()
