#!/usr/bin/env python3
"""Follow a replay run live: print one line per agent launch/finish, order and guard event."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

run = Path(sys.argv[1])
seen: dict[str, int] = {}
names: dict[str, str] = {}


def emit(s: str) -> None:
    print(s[:300], flush=True)


while True:
    for f in sorted((run / "transcripts").glob("*.jsonl")) if (run / "transcripts").exists() else []:
        lines = f.read_text().splitlines()
        start = seen.get(f.name, 0)
        for line in lines[start:]:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = ev.get("message") or {}
            for c in msg.get("content") or [] if isinstance(msg.get("content"), list) else []:
                if c.get("type") == "tool_use":
                    n, inp = c.get("name", ""), c.get("input") or {}
                    if n in ("Agent", "Task"):
                        names[c["id"]] = inp.get("subagent_type", "?")
                        emit(f"[{f.stem}] -> {inp.get('subagent_type')}: {inp.get('description', '')}")
                    elif "place_stock_order" in n or "cancel_order" in n or "close_position" in n or "replace_order" in n:
                        emit(f"[{f.stem}] ORDER CALL {n.split('__')[-1]} {json.dumps(inp)[:200]}")
                    elif "create_alert" in n:
                        emit(f"[{f.stem}] ALERT {inp.get('symbol')} {inp.get('condition')} {inp.get('price')}")
                elif c.get("type") == "tool_result":
                    txt = c.get("content")
                    txt = " ".join(x.get("text", "") for x in txt if isinstance(x, dict)) if isinstance(txt, list) else str(txt)
                    if c.get("tool_use_id") in names:
                        emit(f"[{f.stem}] <- {names.pop(c['tool_use_id'])} done: {txt.strip()[:200]!r}")
                    elif "order guard" in txt:
                        emit(f"[{f.stem}] GUARD {txt.strip()[:220]!r}")
            if ev.get("type") == "result":
                emit(f"[{f.stem}] RESULT turns={ev.get('num_turns')} cost=${ev.get('total_cost_usd')} "
                     f"{(ev.get('result') or '')[:200]!r}")
        seen[f.name] = len(lines)
    if (run / "STOP").exists():
        break
    time.sleep(10)
