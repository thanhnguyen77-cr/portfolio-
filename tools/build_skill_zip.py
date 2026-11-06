#!/usr/bin/env python3
"""Bundle the Trader skill as a standalone skill folder / zip.

The plugin keeps agents in plugins/trader/agents/. A standalone skill (copied
into ~/.claude/skills/, or uploaded to claude.ai as a zip) needs them inside
the skill folder, so this script assembles:

  dist/trader/            SKILL.md, scripts/, references/, templates/, agents/
  dist/trader-skill.zip   the same folder, zipped with trader/ at the top level

  python3 tools/build_skill_zip.py
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "trader"
SKILL = PLUGIN / "skills" / "trader"
DIST = ROOT / "dist"


def main() -> int:
    out = DIST / "trader"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SKILL, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
    (out / "agents").mkdir(exist_ok=True)
    agents = sorted(PLUGIN.glob("agents/trader-*.md"))
    for f in agents:
        shutil.copy2(f, out / "agents" / f.name)
    zpath = DIST / "trader-skill.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(DIST).as_posix())
    print(f"built {out} ({len(agents)} agents) and {zpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
