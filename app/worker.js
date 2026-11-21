/* Trader Desk app worker: hosts Pyodide so the page stays responsive.
   Loads the desk's own Python scripts and the market data, then answers
   requests from the page: info, run (one session), simulate (walk forward). */

const PYODIDE_VERSION = "0.27.2";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
importScripts(PYODIDE_BASE + "pyodide.js");

const PY_FILES = ["traderlib.py", "regime.py", "screen.py", "backtest.py", "kelly.py", "risk_check.py",
                  "engine.py", "desk_web.py"];

const HIST = {
  stocks: ["AAPL", "AMZN", "MSFT", "GOOG", "FB", "INTC", "CSCO", "JPM", "BAC", "V", "MA", "HD", "UNH",
           "BA", "CAT", "DIS", "MCD", "PCLN", "CELG", "XOM"],
  stock: (s) => `https://raw.githubusercontent.com/yumoxu/stocknet-dataset/master/price/raw/${s}.csv`,
  spy: "https://raw.githubusercontent.com/quantopian/zipline/master/zipline/resources/market_data/SPY_benchmark.csv",
  vix: "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
};

let py = null;
const loaded = new Set();
const metas = {};

function post(type, data) { self.postMessage({ type, ...data }); }

async function text(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.text();
}

async function init() {
  if (py) return;
  post("status", { text: "Loading Python" });
  py = await loadPyodide({ indexURL: PYODIDE_BASE });
  py.FS.mkdirTree("/home/pyodide/py");
  for (const f of PY_FILES) {
    py.FS.writeFile(`/home/pyodide/py/${f}`, await text(`py/${f}`));
  }
  await py.runPythonAsync(`
import sys, os
sys.path.insert(0, "/home/pyodide/py")
os.environ["DESK_DATA_ROOT"] = "/data"
import desk_web as D
`);
}

async function loadDataset(ds) {
  if (loaded.has(ds)) return;
  py.FS.mkdirTree(`/data/${ds}`);
  if (ds === "live") {
    post("status", { text: "Loading latest market data" });
    const meta = JSON.parse(await text("data/live/meta.json"));
    metas.live = meta;
    let n = 0;
    await Promise.all(meta.symbols.map(async (s) => {
      py.FS.writeFile(`/data/live/${s}.csv`, await text(`data/live/${s}.csv`));
      post("status", { text: `Loading latest market data (${++n} of ${meta.symbols.length})` });
    }));
  } else {
    post("status", { text: "Loading 2012 to 2017 history from GitHub" });
    let n = 0;
    await Promise.all(HIST.stocks.map(async (s) => {
      py.FS.writeFile(`/data/hist2017/${s}.raw`, await text(HIST.stock(s)));
      post("status", { text: `Loading 2012 to 2017 history (${++n} of ${HIST.stocks.length})` });
    }));
    py.FS.writeFile("/data/hist2017/SPY.returns", await text(HIST.spy));
    py.FS.writeFile("/data/hist2017/VIX.raw", await text(HIST.vix));
    await py.runPythonAsync(`
import csv, io, os, traderlib as tl
d = "/data/hist2017"
for f in os.listdir(d):
    p = f"{d}/{f}"
    if f.endswith(".raw"):
        bars = tl.load_bars(p) if not f.startswith("VIX") else tl.parse_bars(open(p).read(), fmt="csv")
        tl.write_bars_csv(bars, p[:-4] + ".csv")
    elif f == "SPY.returns":
        lv, rows = 100.0, []
        for r in csv.DictReader(io.StringIO(open(p).read())):
            lv *= 1 + float(r["return"])
            rows.append({"time": r["date"][:10], "open": lv, "high": lv, "low": lv, "close": lv, "volume": 0})
        tl.write_bars_csv(rows, f"{d}/SPY.csv")
`);
  }
  loaded.add(ds);
}

async function call(expr) {
  const out = await py.runPythonAsync(expr);
  return JSON.parse(out);
}

self.onmessage = async (e) => {
  const { id, cmd, args } = e.data;
  try {
    await init();
    if (args && args.dataset) await loadDataset(args.dataset);
    py.globals.set("ARGS", JSON.stringify(args || {}));
    let result;
    if (cmd === "info") {
      result = await call(`import json; a = json.loads(ARGS); D.to_json(D.info(a["dataset"]))`);
      result.meta = metas[args.dataset] || null;
    } else if (cmd === "run") {
      post("status", { text: "Running the desk" });
      result = await call(`import json; a = json.loads(ARGS)
D.to_json(D.run_session(a["dataset"], a["asof"], a["settings"]))`);
    } else if (cmd === "analyze") {
      post("status", { text: `Analysing ${args.symbol}` });
      result = await call(`import json; a = json.loads(ARGS)
D.to_json(D.analyze_symbol(a["dataset"], a["asof"], a["symbol"], a["settings"], a.get("setup")))`);
    } else if (cmd === "bars") {
      result = await call(`import json; a = json.loads(ARGS); D.bars_json(a["dataset"], a["symbol"], a["asof"], a.get("n", 120))`);
    } else if (cmd === "simulate") {
      py.globals.set("progress", (n, total, day) => post("progress", { n, total, day }));
      result = await call(`import json; a = json.loads(ARGS)
D.to_json(D.simulate(a["dataset"], a["start"], a["end"], a["every"], a["settings"], progress))`);
    }
    post("result", { id, result });
  } catch (err) {
    post("error", { id, message: String(err && err.message || err).split("\n").slice(-3).join(" ") });
  }
};
