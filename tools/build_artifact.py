#!/usr/bin/env python3
"""Bundle the Trader Desk app as a self-contained Claude artifact.

The artifact sandbox allows no outside fetches, so everything the app needs
is copied next to the page: the Python runtime (Pyodide), the desk's scripts,
both data sets and the recorded AI replay.

  python3 tools/build_artifact.py --pyodide <dir with pyodide.js etc> --live <live data dir> --out <dir>
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins" / "trader" / "skills" / "trader" / "scripts"
PY = ["traderlib.py", "regime.py", "screen.py", "backtest.py", "kelly.py", "risk_check.py"]
PYODIDE_FILES = ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "pyodide-lock.json"]
HIST = ["AAPL", "AMZN", "MSFT", "GOOG", "FB", "INTC", "CSCO", "JPM", "BAC", "V", "MA", "HD", "UNH", "BA", "CAT",
        "DIS", "MCD", "PCLN", "CELG", "XOM", "SPY", "VIX"]


def page(src: str) -> str:
    """Reshape site/app/index.html into an artifact page (no document skeleton, themed tokens, safe gutters)."""
    body = src.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    style = re.search(r"<style>(.*?)</style>", src, re.S).group(1)
    # theme tokens in the shape the artifact viewer expects
    m = re.search(r"@media \(prefers-color-scheme: dark\) \{\s*:root \{(.*?)\}\s*\}", style, re.S)
    dark = m.group(1).strip().rstrip(";")
    style = style.replace(m.group(0),
                          "@media (prefers-color-scheme: dark) { :root:not([data-theme=\"light\"]) { " + dark +
                          "; color-scheme: dark; } }\n:root[data-theme=\"dark\"] { " + dark + "; color-scheme: dark; }")
    # side gutters that clear a phone's system bar; sticky bar below the top inset
    style = style.replace(".bar .in { max-width: 1040px; margin: 0 auto; padding: 0 22px;",
                          ".bar .in { max-width: 1040px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));")
    style = style.replace("main { max-width: 1040px; margin: 0 auto; padding: 0 22px 80px; }",
                          "main { max-width: 1040px; margin: 0 auto; padding-block: 0 80px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }")
    style = style.replace(".bar { position: sticky; top: 0;", ".bar { position: sticky; top: env(safe-area-inset-top, 0px);")
    style += "\nbody::before { content: \"\"; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }\n"
    # routing: plain hash names only, and follow hash changes
    body = body.replace('history.replaceState(null, "", "#" + t);', 'if (location.hash !== "#" + t) location.hash = t;')
    body = body.replace("renderChips();\nconst params = new URLSearchParams(location.search);\nif (params.get(\"ds\") === \"hist2017\") $(\"#dsSeg button[data-ds=hist2017]\").click(); else loadInfo();",
                        "renderChips();\nloadInfo();\nwindow.addEventListener(\"hashchange\", () => { const t = location.hash.slice(1); if (t in HERO && t !== state.tab) showTab(t); });")
    body = body.replace('<a class="brand" href="../">', '<a class="brand" href="https://thanhnguyen77-cr.github.io/portfolio-/">')
    head = ("<title>Trader Desk</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
            "<style>" + style + "</style>\n")
    return head + body


def worker(src: str) -> str:
    src = src.replace('const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;',
                      'const PYODIDE_BASE = new URL("pyodide/", self.location.href).href;')
    src = src.replace('py = await loadPyodide({ indexURL: PYODIDE_BASE });',
                      'py = await loadPyodide({ indexURL: PYODIDE_BASE, stdLibURL: PYODIDE_BASE + "python_stdlib_zip.wasm" });')
    # historical data ships with the artifact, already normalised
    start = src.index("  } else {\n    post(\"status\", { text: \"Loading 2012 to 2017 history from GitHub\" });")
    end = src.index("  loaded.add(ds);")
    src = src[:start] + '''  } else {
    post("status", { text: "Loading 2012 to 2017 history" });
    let n = 0;
    await Promise.all(HIST.stocks.concat(["SPY", "VIX"]).map(async (s) => {
      py.FS.writeFile(`/data/hist2017/${s}.csv`, await text(`data/hist2017/${s}.csv`));
      post("status", { text: `Loading 2012 to 2017 history (${++n} of ${HIST.stocks.length + 2})` });
    }));
  }
''' + src[end:]
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyodide", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "py").mkdir(parents=True)
    (out / "pyodide").mkdir()
    (out / "data" / "live").mkdir(parents=True)
    (out / "data" / "hist2017").mkdir()
    app = ROOT / "site" / "app"
    (out / "index.html").write_text(page((app / "index.html").read_text()))
    (out / "worker.js").write_text(worker((app / "worker.js").read_text()))
    for f in PY:
        shutil.copy2(SCRIPTS / f, out / "py" / f)
    shutil.copy2(ROOT / "sim" / "engine.py", out / "py" / "engine.py")
    shutil.copy2(app / "desk_web.py", out / "py" / "desk_web.py")
    for f in PYODIDE_FILES:
        shutil.copy2(Path(a.pyodide) / f, out / "pyodide" / f)
    # the viewer serves no .zip files, so the standard library ships under a served extension
    shutil.copy2(Path(a.pyodide) / "python_stdlib.zip", out / "pyodide" / "python_stdlib_zip.wasm")
    for f in Path(a.live).iterdir():
        shutil.copy2(f, out / "data" / "live" / f.name)
    for s in HIST:
        shutil.copy2(ROOT / "sim" / "data" / f"{s}.csv", out / "data" / "hist2017" / f"{s}.csv")
    shutil.copy2(app / "data" / "replay2017.json", out / "data" / "replay2017.json")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(files)} files, {total / 1e6:.1f} MB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
