#!/usr/bin/env python3
"""Export a recorded replay run for the app's "AI replay" tab.

  python3 sim/export_replay.py --run replay2017 --out site/app/data/replay2017.json

For every headless Claude Code turn it keeps the desk lead's final message
and, for each agent the desk lead briefed, the brief and the agent's own
final report (its last message in the transcript). Paths are shortened and
long texts trimmed.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def clean(s: str, limit: int = 9000) -> str:
    s = s or ""
    s = re.sub(r"/home/user/portfolio-/sim/runs/[^/\s`]+/desk/", "", s)
    s = re.sub(r"/home/user/portfolio-/", "", s)
    s = re.sub(r"<tools-used-before-summary>.*?</tools-used-before-summary>", "", s, flags=re.S)
    return s.strip()[:limit]


def turn_agents(path: Path) -> list[dict]:
    agents: dict[str, dict] = {}
    order: list[str] = []
    last_text: dict[str, str] = {}
    results: dict[str, str] = {}
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") or {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        parent = ev.get("parent_tool_use_id")
        if ev.get("type") == "assistant" and parent:
            txt = "\n".join(c.get("text", "") for c in content if c.get("type") == "text").strip()
            if txt:
                last_text[parent] = txt
        for c in content:
            if c.get("type") == "tool_result" and not parent:
                cc = c.get("content")
                t = cc if isinstance(cc, str) else "\n".join(x.get("text", "") for x in cc or [] if isinstance(x, dict))
                if "The report follows:" in t:
                    body = t.split("The report follows:", 1)[1]
                    results[c.get("tool_use_id")] = "\n".join(l[2:] if l.startswith("  ") else l for l in body.splitlines())
            if c.get("type") == "tool_use" and c.get("name") in ("Agent", "Task"):
                inp = c.get("input") or {}
                agents[c["id"]] = {"agent": str(inp.get("subagent_type", "?")).split(":")[-1],
                                   "description": inp.get("description", ""), "brief": clean(inp.get("prompt", ""), 5000)}
                order.append(c["id"])
    out = []
    for k in order:
        a = agents[k]
        a["report"] = clean(last_text.get(k) or results.get(k) or "(no report captured)")
        out.append(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    sessions = json.loads((run / "sessions.json").read_text())
    st = json.loads((run / "state.json").read_text())
    ws = run / "desk" / "trader-workspace"
    out_sessions = []
    for s in sessions:
        day = s["day"]
        chips = []
        if s["turn"] == 1:
            reg = ws / "regime" / f"{day}.json"
            if reg.exists():
                r = json.loads(reg.read_text())
                chips.append(f"regime {r['regime'].replace('_', ' ')}")
            scr = ws / "screens" / f"{day}.json"
            if scr.exists():
                r = json.loads(scr.read_text())
                chips.append(f"{r['passed_count']} of {r['universe']} passed")
            for p in sorted((ws / "backtests").glob(f"*-{day}.json")):
                b = json.loads(p.read_text())
                chips.append(f"{p.stem.split('-')[0]} backtest {b.get('verdict')}")
        placed = [l for l in st.get("log", []) if l["event"] == "order_placed" and l["asof"] == day]
        if s["kind"] == "confirm":
            chips += [f"order: buy {l['detail']['qty']} {l['detail']['symbol']}" for l in placed]
        kind = {"day": "trading session", "confirm": "you said yes", "review": "journal review"}[s["kind"]]
        out_sessions.append({
            "day": day, "day_label": date.fromisoformat(day).strftime("%a %d %b %Y"), "kind": kind,
            "minutes": round((s.get("seconds") or 0) / 60), "chips": chips,
            "prompt": s["prompt"].split("\n\n", 1)[-1] if s["turn"] == 1 else s["prompt"],
            "result": clean(s.get("result") or "", 12000),
            "agents": turn_agents(run / s["transcript"]),
        })
    eq = st["equity_history"][-1]["equity"] if st.get("equity_history") else None
    data = {
        "intro": ("The real Claude Code desk (Opus 5.5, all eleven agents, the scripts and the order-guard hook) ran "
                  "headless on real daily prices from 2017. Replay connectors served data only up to each session's "
                  "date. The harness gave the money rules ($10,000 budget, $2,000 kept aside, $2,500 max per stock) "
                  "and said yes to each order card. A paper broker filled orders on the real prices that followed."),
        "sessions": out_sessions,
        "orders": len([o for o in st.get("orders", []) if not o.get("parent_id")]),
        "fills": st.get("fills", []),
        "end_equity": eq, "asof": st.get("asof"),
        "cost_usd": round(sum({s.get("session_id"): s.get("cost_usd") or 0 for s in sessions}.values()), 2),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=1))
    print(f"wrote {a.out}: {len(out_sessions)} turns, {sum(len(x['agents']) for x in out_sessions)} agent runs")


if __name__ == "__main__":
    main()
