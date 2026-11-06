#!/usr/bin/env python3
"""Run the real Trader plugin, headless, through a historical replay.

For each session date the harness:
  1. advances the replay broker to that date's close (filling orders on the
     real bars in between and firing alerts);
  2. starts a fresh headless Claude Code session in the desk folder, with the
     trader plugin loaded from this repo and the simulated TradingView and
     Alpaca connectors as its only MCP servers;
  3. asks the desk lead to run the day's trading loop (the first session also
     says "install skill" and gives the money rules);
  4. plays the human: if the desk shows an order card, it answers
     "Yes, place it exactly as shown." in the same session. The risk manager,
     approval tickets and order-guard hook still apply.
A final session runs the journal review. Every turn is saved as stream-json.

  python3 sim/run_sim.py init  --run may2017
  python3 sim/run_sim.py day   --run may2017 --date 2017-05-05
  python3 sim/run_sim.py review --run may2017 --date 2017-09-01
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PLUGIN = REPO / "plugins" / "trader"
sys.path.insert(0, str(HERE))

WATCHLIST = ["NASDAQ:AAPL", "NASDAQ:AMZN", "NASDAQ:MSFT", "NASDAQ:GOOG", "NASDAQ:FB", "NASDAQ:INTC",
             "NASDAQ:CSCO", "NYSE:JPM", "NYSE:BAC", "NYSE:V", "NYSE:HD", "NYSE:UNH", "NYSE:BA", "NYSE:CAT"]
MONEY_RULES = "budget $10,000, keep $2,000 aside as cash, max $2,500 in any one stock"
START_CASH = 10_000.0

ALLOWED = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "Task", "Skill", "ToolSearch",
           "TodoWrite", "mcp__alpaca", "mcp__tradingview"]
DISALLOWED = ["WebSearch", "WebFetch"]  # the web would leak post-replay information
SCRUB_ENV = ["CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_REMOTE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET",
             "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_TEE_SDK_STDOUT", "CLAUDE_CODE_DIAGNOSTICS_FILE"]


def run_dir(name: str) -> Path:
    return HERE / "runs" / name


def set_env(rd: Path) -> None:
    os.environ["SIM_STATE"] = str(rd / "state.json")
    os.environ["SIM_DATA"] = str(HERE / "data")
    os.environ["SIM_DESK"] = str(rd / "desk")


def nice(d: str) -> str:
    return date.fromisoformat(d).strftime("%A %d %B %Y")


PREAMBLE = """[HISTORICAL REPLAY - paper simulation]
Today is {day}, after the US market close. Treat {iso} as today's date everywhere: in file names, as_of fields and when counting days. The TradingView and Alpaca connectors in this session replay real historical market data and only contain data up to today's close; nothing later exists yet. Do not use any knowledge you may have of what happened in markets after today: every decision must come from the connectors and the desk's scripts. There is no news feed in this replay. Tell every agent you brief that this is a replay dated {iso}.
"""

FIRST_DAY = """install skill

My money rules: {rules}. Use the desk defaults for everything else. The Alpaca connector is the paper account (ALPACA_PAPER_TRADE=true).

Then run today's full trading loop on my TradingView watchlist "Swing 2017". If a trade clears every gate, show me the confirmation card and stop there; I'll answer."""

NEXT_DAY = """Start by having the journal sync fills, open positions and orders with the broker, and tell me what happened since the last session (fills, stops, targets, alerts). Check open positions against their plans (time stops, invalidation) and handle any exits through the risk manager.

Then run today's trading loop on my TradingView watchlist "Swing 2017", skipping symbols we already hold or have orders in. If a trade clears every gate, show me the confirmation card and stop there; I'll answer."""

REVIEW = """Run the journal: sync every fill and close with the broker, then write the performance review (open positions, closed trades in R, win rate, expectancy, comparison with the backtests, lessons). Give me the headline results."""

CONFIRM = "Yes, place it exactly as shown."


def claude(rd: Path, prompt: str, label: str, session_id: str, resume: bool, model: str) -> dict:
    tdir = rd / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"{label}.jsonl"
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose", "--model", model,
           "--plugin-dir", str(PLUGIN), "--strict-mcp-config", "--mcp-config", str(rd / "mcp.json"),
           "--permission-mode", "acceptEdits", "--allowedTools", *ALLOWED, "--disallowedTools", *DISALLOWED]
    cmd += ["--resume", session_id] if resume else ["--session-id", session_id]
    env = {k: v for k, v in os.environ.items() if k not in SCRUB_ENV}
    t0 = time.time()
    with open(out, "w") as fh:
        proc = subprocess.run(cmd, cwd=rd / "desk", env=env, stdout=fh, stderr=subprocess.PIPE, text=True)
    result = {}
    for line in out.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
    rec = {"label": label, "prompt": prompt, "transcript": str(out.relative_to(rd)), "exit": proc.returncode,
           "stderr": proc.stderr[-2000:], "seconds": round(time.time() - t0), "session_id": session_id,
           "result": result.get("result"), "cost_usd": result.get("total_cost_usd"),
           "num_turns": result.get("num_turns"), "is_error": result.get("is_error")}
    print(f"[{label}] exit={proc.returncode} {rec['seconds']}s turns={rec['num_turns']} cost=${rec['cost_usd']}",
          flush=True)
    return rec


def wants_confirmation(text: str | None) -> bool:
    t = (text or "").lower()
    return "paper order" in t or ("confirm" in t and ("bracket" in t or "stop loss" in t or "stop-loss" in t))


def save_session(rd: Path, rec: dict) -> None:
    p = rd / "sessions.json"
    data = json.loads(p.read_text()) if p.exists() else []
    data.append(rec)
    p.write_text(json.dumps(data, indent=1))


def cmd_init(a) -> None:
    rd = run_dir(a.run)
    (rd / "desk").mkdir(parents=True, exist_ok=True)
    set_env(rd)
    import engine as E
    E.init_state(a.start, START_CASH, {"Swing 2017": WATCHLIST})
    srv = {"command": sys.executable, "env": {k: os.environ[k] for k in ("SIM_STATE", "SIM_DATA", "SIM_DESK")}}
    mcp = {"mcpServers": {"alpaca": {**srv, "args": [str(HERE / "alpaca_server.py")]},
                          "tradingview": {**srv, "args": [str(HERE / "tradingview_server.py")]}}}
    (rd / "mcp.json").write_text(json.dumps(mcp, indent=1))
    (rd / "config.json").write_text(json.dumps({"start": a.start, "model": a.model, "watchlist": WATCHLIST,
                                                "money_rules": MONEY_RULES, "start_cash": START_CASH}, indent=1))
    print(f"initialised {rd} at {a.start}")


def cmd_session(a, kind: str) -> None:
    rd = run_dir(a.run)
    set_env(rd)
    import engine as E
    cfg = json.loads((rd / "config.json").read_text())
    with E.state(write=False) as st:
        cur = st["asof"]
    if a.date > cur:
        print(f"advancing replay {cur} -> {a.date}: {E.advance(a.date)}", flush=True)
    elif a.date < cur:
        sys.exit(f"replay is already at {cur}")
    sid = str(uuid.uuid4())
    pre = PREAMBLE.format(day=nice(a.date), iso=a.date)
    if kind == "review":
        body = REVIEW
    else:
        first = not (rd / "sessions.json").exists()
        body = FIRST_DAY.format(rules=MONEY_RULES) if first else NEXT_DAY
    rec = claude(rd, pre + "\n" + body, f"{a.date}-1-{kind}", sid, False, cfg["model"])
    rec.update(day=a.date, kind=kind, turn=1)
    save_session(rd, rec)
    if kind == "day" and wants_confirmation(rec["result"]):
        rec2 = claude(rd, CONFIRM, f"{a.date}-2-confirm", sid, True, cfg["model"])
        rec2.update(day=a.date, kind="confirm", turn=2)
        save_session(rd, rec2)
    elif kind == "day":
        print(f"[{a.date}] no order card shown; nothing to confirm", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--run", required=True)
    i.add_argument("--start", default="2017-05-05")
    i.add_argument("--model", default="claude-opus-5-5")
    for name in ("day", "review", "advance"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--date", required=True)
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init(a)
    elif a.cmd == "advance":
        rd = run_dir(a.run)
        set_env(rd)
        import engine as E
        print(E.advance(a.date))
    else:
        cmd_session(a, a.cmd)


if __name__ == "__main__":
    main()
