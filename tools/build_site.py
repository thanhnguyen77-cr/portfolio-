#!/usr/bin/env python3
"""Assemble the GitHub Pages site into one folder.

  python3 tools/build_site.py --out _site [--live path/to/live-data]

  _site/                     landing page (site/index.html)
  _site/trader-skill.zip     standalone skill bundle
  _site/app/                 the in-browser desk (site/app)
  _site/app/py/              the desk's Python scripts, loaded by Pyodide
  _site/app/data/live/       latest daily bars, if a live-data folder is given
  _site/jetski/              Turbo Tide, the 3D jet ski racing game (site/jetski)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins" / "trader" / "skills" / "trader" / "scripts"
PY = ["traderlib.py", "regime.py", "screen.py", "backtest.py", "kelly.py", "risk_check.py"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_site")
    ap.add_argument("--live", help="folder with live CSVs + meta.json")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_skill_zip.py")], check=True)
    shutil.copy2(ROOT / "dist" / "trader-skill.zip", out / "trader-skill.zip")
    py = out / "app" / "py"
    py.mkdir(parents=True, exist_ok=True)
    for f in PY:
        shutil.copy2(SCRIPTS / f, py / f)
    shutil.copy2(ROOT / "sim" / "engine.py", py / "engine.py")
    shutil.move(str(out / "app" / "desk_web.py"), py / "desk_web.py")
    if a.live and Path(a.live, "meta.json").exists():
        shutil.copytree(a.live, out / "app" / "data" / "live", dirs_exist_ok=True)
        print(f"live data: {len(list(Path(a.live).glob('*.csv')))} files")
    else:
        print("live data: none")
    (out / ".nojekyll").touch()
    print(f"site assembled in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
