#!/usr/bin/env python3
"""Install the Trader desk: agents, order-guard hook and workspace.

What "install skill" does:
  1. copies the 11 trader-* agent definitions into ~/.claude/agents
     (skipped when the skill was installed as a Claude Code plugin, which
     registers the agents itself)
  2. registers the order-guard PreToolUse hook in .claude/settings.json of
     the current project (skipped for plugin installs, which ship the hook)
  3. creates ./trader-workspace with data/, screens/, plans/, backtests/,
     sizing/, news/, alerts/, risk/approvals/, orders/ and reports/
  4. writes trader-workspace/paths.json so every agent can find the scripts
  5. prints which connectors (TradingView, Alpaca) still need to be added

  install.py                      # auto-detect plugin vs. standalone
  install.py --agents project     # put agents in ./.claude/agents instead
  install.py --agents skip --hook skip
  install.py --check              # report status only, change nothing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import traderlib as tl

AGENT_PREFIX = "trader-"
WORKSPACE_DIRS = ("data", "screens", "plans", "backtests", "sizing", "news", "alerts",
                  "risk/approvals", "orders", "reports", "regime")
HOOK_MATCHER = "mcp__.*alpaca.*"


def agents_source() -> Optional[Path]:
    for cand in (tl.SKILL_DIR / "agents", tl.SKILL_DIR.parent.parent / "agents"):
        if cand.is_dir() and any(cand.glob(f"{AGENT_PREFIX}*.md")):
            return cand
    return None


def is_plugin_install() -> bool:
    parts = tl.SKILL_DIR.as_posix()
    return "/.claude/plugins/" in parts


def install_agents(dest_kind: str, check: bool) -> str:
    src = agents_source()
    if src is None:
        return "agents: source folder not found next to the skill — skipped"
    files = sorted(src.glob(f"{AGENT_PREFIX}*.md"))
    dest = (Path.home() / ".claude" / "agents") if dest_kind == "user" else (Path.cwd() / ".claude" / "agents")
    if check:
        present = [f.name for f in files if (dest / f.name).exists()]
        return f"agents: {len(present)}/{len(files)} present in {dest}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest / f.name)
    return f"agents: installed {len(files)} → {dest} ({', '.join(f.stem for f in files)})"


def install_hook(kind: str, check: bool) -> str:
    settings = (Path.home() / ".claude" / "settings.json") if kind == "user" else (Path.cwd() / ".claude" / "settings.json")
    cmd = f'python3 "{tl.SCRIPTS_DIR / "order_guard.py"}"'
    data: dict = {}
    if settings.exists():
        try:
            data = json.loads(settings.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError:
            return f"hook: {settings} is not valid JSON — add the hook by hand (see references/setup.md)"
    pre = data.setdefault("hooks", {}).setdefault("PreToolUse", [])
    already = any("order_guard.py" in h.get("command", "") for e in pre for h in e.get("hooks", []))
    if check:
        return f"hook: order guard {'registered' if already else 'NOT registered'} in {settings}"
    if already:
        return f"hook: order guard already registered in {settings}"
    pre.append({"matcher": HOOK_MATCHER, "hooks": [{"type": "command", "command": cmd, "timeout": 15}]})
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return f"hook: order guard registered in {settings} (matcher {HOOK_MATCHER})"


def setup_workspace(ws: Path, mode: str, check: bool) -> list[str]:
    msgs = []
    if check:
        msgs.append(f"workspace: {'exists' if ws.exists() else 'missing'} at {ws}")
    else:
        for d in WORKSPACE_DIRS:
            (ws / d).mkdir(parents=True, exist_ok=True)
        tl.dump_json({"scripts_dir": str(tl.SCRIPTS_DIR), "skill_dir": str(tl.SKILL_DIR),
                      "workspace": str(ws), "install_mode": mode, "installed_at": tl.now_iso()},
                     ws / "paths.json")
        gi = ws / ".gitignore"
        if not gi.exists():
            gi.write_text("# trading data, account snapshots and approvals stay local\n*\n!.gitignore\n",
                          encoding="utf-8")
        msgs.append(f"workspace: ready at {ws}")
    prof = ws / "profile.json"
    if prof.exists():
        errs = tl.validate_profile(tl.merge_defaults(tl.load_json(prof)))
        msgs.append("profile: OK" if not errs else "profile: INVALID — " + "; ".join(errs))
    else:
        msgs.append("profile: not set yet — next, tell the desk your budget, cash to keep aside and max per stock")
    return msgs


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", choices=["auto", "user", "project", "skip"], default="auto")
    ap.add_argument("--hook", choices=["auto", "project", "user", "skip"], default="auto")
    ap.add_argument("--workspace")
    ap.add_argument("--check", action="store_true", help="report only; change nothing")
    args = ap.parse_args(argv)

    plugin = is_plugin_install()
    mode = "plugin" if plugin else "standalone"
    agents = args.agents if args.agents != "auto" else ("skip" if plugin else "user")
    hook = args.hook if args.hook != "auto" else ("skip" if plugin else "project")
    ws = tl.workspace_dir(args.workspace)

    lines = [f"Trader desk — {'status' if args.check else 'install'} ({mode})", ""]
    if agents == "skip":
        lines.append("agents: provided by the plugin" if plugin else "agents: skipped")
    else:
        lines.append(install_agents(agents, args.check))
    if hook == "skip":
        lines.append("hook: order guard provided by the plugin" if plugin else "hook: skipped (NOT recommended)")
    else:
        lines.append(install_hook(hook, args.check))
    lines += setup_workspace(ws, mode, args.check)
    lines += [
        "",
        "Connectors (add these in Claude settings / `claude mcp add`):",
        "  [ ] TradingView — charts, watchlists, price history, alerts",
        "  [ ] Alpaca (paper) — account, positions, orders; also a fallback for price history and news",
        "Restart Claude Code (or run /agents) so newly installed agents load.",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
