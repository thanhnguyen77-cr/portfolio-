#!/usr/bin/env python3
"""Build an HTML report (and screenshots) for a replay run.

  python3 sim/report.py --run replay2017 [--shots]

Sources: the replay broker's state (ground truth for orders, fills, equity),
the desk's own workspace files, and the stream-json transcripts of every
headless Claude Code turn (agent briefs and reports, tool calls, guard events).
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ----------------------------------------------------------- markdown-lite

def md(text: str) -> str:
    """Small Markdown renderer: headings, lists, tables, code fences, bold/italic/code."""
    lines = (text or "").replace("\r", "").split("\n")
    out: list[str] = []
    i = 0

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", s)
        s = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<i>\1</i>", s)
        return s

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            out.append("<pre class='code'>" + html.escape("\n".join(buf)) + "</pre>")
            i += 1
            continue
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            head = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = "<div class='tw'><table><thead><tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(t + "</tbody></table></div>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)", ln)
        if m:
            lvl = min(len(m.group(1)) + 2, 6)
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if re.match(r"^\s*([-*•]|\d+[.)])\s+", ln):
            ordered = bool(re.match(r"^\s*\d+[.)]\s+", ln))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines) and re.match(r"^\s*([-*•]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*•]|\d+[.)])\s+", "", lines[i]))
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if ln.strip() in ("---", "***"):
            out.append("<hr>")
            i += 1
            continue
        if ln.strip().startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue
        if not ln.strip():
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,6}\s|\s*([-*•]|\d+[.)])\s|\s*\||\s*```|\s*>)", lines[i]):
            buf.append(lines[i])
            i += 1
        if not buf:
            buf.append(lines[i])
            i += 1
        out.append("<p>" + "<br>".join(inline(x) for x in buf) + "</p>")
    return "\n".join(out)


# ----------------------------------------------------------- transcripts

def parse_transcript(path: Path) -> dict:
    agents: dict[str, dict] = {}
    order: list[str] = []
    calls: list[dict] = []
    guard: list[str] = []
    result: dict = {}
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            result = ev
        msg = ev.get("message") or {}
        content = msg.get("content") if isinstance(msg.get("content"), list) else []
        for c in content:
            if c.get("type") == "tool_use":
                n, inp = c.get("name", ""), c.get("input") or {}
                if n in ("Agent", "Task"):
                    agents[c["id"]] = {"agent": str(inp.get("subagent_type", "?")).split(":")[-1],
                                       "description": inp.get("description", ""), "brief": inp.get("prompt", ""),
                                       "report": None}
                    order.append(c["id"])
                elif n.startswith("mcp__"):
                    calls.append({"tool": n, "input": inp, "parent": ev.get("parent_tool_use_id")})
            elif c.get("type") == "tool_result":
                txt = c.get("content")
                txt = "\n".join(x.get("text", "") for x in txt if isinstance(x, dict)) if isinstance(txt, list) else str(txt or "")
                if c.get("tool_use_id") in agents:
                    agents[c["tool_use_id"]]["report"] = txt
                if "order guard" in txt:
                    guard.append(txt.strip())
    return {"agents": [agents[k] for k in order], "calls": calls, "guard": guard, "result": result}


# ----------------------------------------------------------- charts

def svg_lines(series: list[dict], width=960, height=280, pad=44, fmt=lambda v: f"{v:,.0f}") -> str:
    """series: [{name, color, points:[(label, value)], dash?}] sharing x labels."""
    allv = [v for s in series for _, v in s["points"] if v is not None]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    if hi == lo:
        hi += 1
    span = hi - lo
    lo -= span * 0.08
    hi += span * 0.08
    n = max(len(s["points"]) for s in series)
    xw = (width - pad * 2) / max(1, n - 1)
    y = lambda v: pad / 2 + (hi - v) / (hi - lo) * (height - pad * 1.5)  # noqa: E731
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>"]
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        yy = y(v)
        parts.append(f"<line x1='{pad}' x2='{width - pad / 2}' y1='{yy:.1f}' y2='{yy:.1f}' class='grid'/>"
                     f"<text x='{pad - 6}' y='{yy + 4:.1f}' class='ax' text-anchor='end'>{fmt(v)}</text>")
    labels = series[0]["points"]
    step = max(1, len(labels) // 6)
    for idx in range(0, len(labels), step):
        parts.append(f"<text x='{pad + idx * xw:.1f}' y='{height - 6}' class='ax' text-anchor='middle'>{labels[idx][0][5:]}</text>")
    for s in series:
        pts = " ".join(f"{pad + i * xw:.1f},{y(v):.1f}" for i, (_, v) in enumerate(s["points"]) if v is not None)
        dash = " stroke-dasharray='5 4'" if s.get("dash") else ""
        parts.append(f"<polyline points='{pts}' fill='none' stroke='{s['color']}' stroke-width='{s.get('w', 2)}'{dash}/>")
    for s in series:
        for (lab, v, txt, col) in s.get("marks", []):
            idx = next((i for i, (l2, _) in enumerate(s["points"]) if l2 == lab), None)
            if idx is None:
                continue
            cx, cy = pad + idx * xw, y(v)
            parts.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' fill='{col}' stroke='var(--bg)' stroke-width='2'/>"
                         f"<text x='{cx + 8:.1f}' y='{cy - 8:.1f}' class='mk' fill='{col}'>{html.escape(txt)}</text>")
    for (v, txt, col) in series[0].get("levels", []):
        yy = y(v)
        parts.append(f"<line x1='{pad}' x2='{width - pad / 2}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='{col}' stroke-dasharray='6 4' stroke-width='1.4'/>"
                     f"<text x='{width - pad / 2 - 4}' y='{yy - 5:.1f}' class='mk' text-anchor='end' fill='{col}'>{html.escape(txt)}</text>")
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------- build

def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def money(v) -> str:
    return "—" if v is None else f"${v:,.2f}"


def build(run: Path) -> Path:
    import engine as E  # noqa: F401  (uses SIM_* env set by caller)
    cfg = load_json(run / "config.json") or {}
    sessions = load_json(run / "sessions.json") or []
    st = load_json(run / "state.json") or {}
    ws = run / "desk" / "trader-workspace"
    eqh = st.get("equity_history", [])
    start_eq = cfg.get("start_cash", 10000.0)
    end_eq = eqh[-1]["equity"] if eqh else start_eq
    dates = [h["date"] for h in eqh]
    spy = {d: E.close_on("SPY", d) for d in dates}
    spy0 = spy.get(dates[0]) if dates else None
    peak, mdd = 0.0, 0.0
    for h in eqh:
        peak = max(peak, h["equity"])
        mdd = max(mdd, (peak - h["equity"]) / peak if peak else 0)
    spy_ret = (spy[dates[-1]] / spy0 - 1) * 100 if dates and spy0 else None
    cost = sum(s.get("cost_usd") or 0 for s in sessions)
    secs = sum(s.get("seconds") or 0 for s in sessions)

    # orders / trades from broker truth
    orders = st.get("orders", [])
    by_id = {o["id"]: o for o in orders}
    trades = []
    for o in orders:
        if o.get("parent_id") or o["side"] != "buy":
            continue
        legs = [by_id[l] for l in o.get("legs") or [] if l in by_id]
        tp = next((l for l in legs if l.get("leg") == "take_profit"), None)
        sl = next((l for l in legs if l.get("leg") == "stop_loss"), None)
        exit_leg = next((l for l in legs if l["status"] == "filled"), None)
        manual = [f for f in st.get("fills", []) if f["symbol"] == o["symbol"] and f["kind"] == "exit"
                  and o.get("filled_at") and f["date"] >= o["filled_at"]]
        exit_px = exit_leg["filled_avg_price"] if exit_leg else (manual[0]["price"] if manual else None)
        exit_day = exit_leg["filled_at"] if exit_leg else (manual[0]["date"] if manual else None)
        exit_kind = exit_leg.get("leg") if exit_leg else ("manual exit" if manual else None)
        entry = o.get("filled_avg_price")
        stop = sl["stop_price"] if sl else None
        last = E.close_on(o["symbol"], st["asof"])
        pnl = None
        r = None
        if entry:
            ref = exit_px if exit_px else last
            pnl = (ref - entry) * o["qty"]
            r = (ref - entry) / (entry - stop) if stop and entry > stop else None
        trades.append({"o": o, "tp": tp, "sl": sl, "entry": entry, "exit_px": exit_px, "exit_day": exit_day,
                       "exit_kind": exit_kind, "pnl": pnl, "r": r, "open": bool(entry and not exit_px), "last": last})

    # ---------------------------------------------------------- html
    H: list[str] = []
    H.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Trader Desk Replay</title>
<style>{CSS}</style></head><body><main class="wrap">""")
    first = sessions[0]["day"] if sessions else cfg.get("start")
    last_day = st.get("asof")
    H.append(f"""<header class="hero" id="top"><div class="eyebrow">HISTORICAL REPLAY · REAL AGENTS · SIMULATED PAPER BROKER</div>
<h1>Trader desk, {first} → {last_day}</h1>
<p class="lede">The real Trader plugin (11 agents, scripts and order-guard hook) ran headless in Claude Code ({html.escape(cfg.get('model', ''))}).
Its TradingView and Alpaca connectors were replaced by replay connectors that serve real daily market data only up to each session's date.
The harness played the human: it gave the money rules ({html.escape(cfg.get('money_rules', ''))}) and answered "yes" to every order card the desk showed.
It never placed or changed an order itself.</p></header>""")
    ntr = len([t for t in trades if t["entry"]])
    closed = [t for t in trades if t["exit_px"]]
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    H.append("<section id='kpis'><div class='kpis'>")
    for lab, val, sub in [
        ("Ending equity", money(end_eq), f"start {money(start_eq)}"),
        ("Desk return", f"{(end_eq / start_eq - 1) * 100:+.2f}%", f"max drawdown {mdd * 100:.2f}%"),
        ("SPY (total return) same window", f"{spy_ret:+.2f}%" if spy_ret is not None else "—", "buy and hold, fully invested"),
        ("Orders → fills", f"{len([t for t in trades])} → {ntr}", f"{len(closed)} closed · {len(wins)} winners"),
        ("Sessions", str(len({s['day'] for s in sessions})), f"{len(sessions)} turns · {secs // 60} min"),
        ("Model cost", f"${cost:,.2f}" if cost else "—", "sum of reported session costs"),
    ]:
        H.append(f"<div class='kpi'><span>{lab}</span><b>{val}</b><small>{sub}</small></div>")
    H.append("</div>")
    if eqh:
        pts = [(h["date"], h["equity"]) for h in eqh]
        spy_pts = [(d, start_eq * spy[d] / spy0 if spy0 and spy.get(d) else None) for d in dates]
        marks = []
        for f in st.get("fills", []):
            col = "var(--up)" if f["kind"] in ("entry",) else ("var(--down)" if f["kind"] == "stop_loss" else "var(--accent)")
            eqd = next((h["equity"] for h in eqh if h["date"] == f["date"]), None)
            if eqd:
                marks.append((f["date"], eqd, f"{f['kind'].replace('_', ' ')} {f['symbol']}", col))
        H.append("<h2>Equity vs SPY</h2>" + svg_lines([
            {"name": "Desk equity", "color": "var(--accent)", "points": pts, "w": 2.6, "marks": marks},
            {"name": "SPY (scaled)", "color": "var(--muted)", "points": spy_pts, "dash": True},
        ], fmt=lambda v: f"${v:,.0f}"))
        H.append("<p class='legend'><span class='sw a'></span>Desk equity (cash + positions at each close) "
                 "<span class='sw m'></span>SPY total-return index scaled to the same start</p>")
    H.append("</section>")

    # trades
    H.append("<section id='trades'><h2>Orders and outcomes (broker ground truth)</h2>")
    if trades:
        H.append("<div class='tw'><table><thead><tr><th>Placed</th><th>Symbol</th><th>Entry order</th><th>Qty</th>"
                 "<th>Stop / Target</th><th>Status</th><th>Filled</th><th>Exit</th><th>P&amp;L</th><th>R</th></tr></thead><tbody>")
        for t in trades:
            o = t["o"]
            ent = (f"{o['type']} {o.get('stop_price') or ''}{' / ' if o.get('stop_price') and o.get('limit_price') else ''}"
                   f"{o.get('limit_price') or ''} · {o['time_in_force']}")
            status = o["status"] + (" (open)" if t["open"] else "")
            filled = f"{o.get('filled_at')} @ {o.get('filled_avg_price')}" if o.get("filled_at") else "—"
            ex = f"{t['exit_day']} @ {t['exit_px']} ({t['exit_kind']})" if t["exit_px"] else (f"open, last {t['last']}" if t["entry"] else "—")
            pnl = "—" if t["pnl"] is None else f"<span class='{'up' if t['pnl'] > 0 else 'down'}'>{t['pnl']:+,.2f}</span>" + ("" if t["exit_px"] else " <small>unrealised</small>")
            r = "—" if t["r"] is None else f"{t['r']:+.2f}"
            H.append(f"<tr><td>{o.get('submitted_at')}</td><td><b>{o['symbol']}</b></td><td>{ent}</td><td>{o['qty']}</td>"
                     f"<td>{t['sl']['stop_price'] if t['sl'] else '—'} / {t['tp']['limit_price'] if t['tp'] else '—'}</td>"
                     f"<td>{status}</td><td>{filled}</td><td>{ex}</td><td>{pnl}</td><td>{r}</td></tr>")
        H.append("</tbody></table></div>")
    else:
        H.append("<p class='muted'>The desk placed no orders in this replay.</p>")
    # charts per traded symbol
    for t in trades:
        o = t["o"]
        sym = o["symbol"]
        start = o.get("submitted_at")
        bs = [b for b in E.bars(sym, st["asof"]) if b["time"] >= (E.prev_day(start) or start)]
        pre = [b for b in E.bars(sym, start)][-30:]
        seq = pre + [b for b in bs if b["time"] > start]
        pts = [(b["time"], b["close"]) for b in seq]
        levels = []
        if t["tp"]:
            levels.append((t["tp"]["limit_price"], f"target {t['tp']['limit_price']}", "var(--up)"))
        if t["sl"]:
            levels.append((t["sl"]["stop_price"], f"stop {t['sl']['stop_price']}", "var(--down)"))
        if o.get("limit_price") or o.get("stop_price"):
            lv = o.get("stop_price") or o.get("limit_price")
            levels.append((lv, f"entry {lv}", "var(--accent)"))
        marks = [(f["date"], f["price"], f"{f['kind'].replace('_', ' ')} {f['price']}",
                  "var(--down)" if f["kind"] == "stop_loss" else ("var(--up)" if f["kind"] == "take_profit" else "var(--accent)"))
                 for f in st.get("fills", []) if f["symbol"] == sym and f["date"] >= start]
        marks.insert(0, (start, next((v for d, v in pts if d == start), pts[0][1] if pts else 0), "plan date", "var(--muted)"))
        H.append(f"<div class='card'><h3>{sym} — placed {start}</h3>" + svg_lines(
            [{"name": sym, "color": "var(--text)", "points": pts, "levels": levels, "marks": marks, "w": 1.8}],
            height=240, fmt=lambda v: f"{v:,.2f}") + "</div>")
    H.append("</section>")

    # sessions
    H.append("<section id='sessions'><h2>Session by session: what the desk decided</h2>")
    for day in sorted({s["day"] for s in sessions}):
        H.append(f"<article class='session' id='s-{day}'><h3 class='day'>{day}</h3>")
        reg = load_json(ws / "regime" / f"{day}.json")
        chips = []
        if reg:
            chips.append(f"<span class='chip'>regime <b>{reg.get('regime')}</b> {reg.get('points')}/{reg.get('max_points')}</span>")
        scr = load_json(ws / "screens" / f"{day}.json")
        if scr:
            chips.append(f"<span class='chip'>screen {scr.get('passed_count')}/{scr.get('universe')} passed</span>")
        for p in sorted((ws / "backtests").glob(f"*{day}*.json")) if (ws / "backtests").exists() else []:
            b = load_json(p) or {}
            chips.append(f"<span class='chip {'ok' if b.get('verdict') == 'PASS' else 'bad'}'>backtest {', '.join(b.get('symbols', [])[:1])} <b>{b.get('verdict')}</b></span>")
        for p in sorted((ws / "sizing").glob(f"*{day}*.json")) if (ws / "sizing").exists() else []:
            z = load_json(p) or {}
            chips.append(f"<span class='chip'>size {p.stem.split('-')[0]} <b>{z.get('decision')} {z.get('qty', '')}</b> ({z.get('binding_constraint')})</span>")
        for p in sorted((ws / "risk").glob("*.json")) if (ws / "risk").exists() else []:
            rk = load_json(p) or {}
            if day.replace("-", "") in p.stem.replace("-", "") or rk.get("generated_at", "")[:10] == day:
                chips.append(f"<span class='chip {'ok' if rk.get('decision') == 'APPROVE' else 'bad'}'>risk {(rk.get('order') or {}).get('symbol')} <b>{rk.get('decision')}</b></span>")
        if chips:
            H.append("<div class='chips'>" + "".join(chips) + "</div>")
        if scr and scr.get("results"):
            H.append("<details><summary>Screener table</summary><div class='tw'><table><thead><tr><th>Symbol</th><th>Pass</th><th>Score</th><th>Setup</th><th>RSI14</th><th>ATR%</th><th>Rel vol</th><th>3m %</th><th>Failed</th></tr></thead><tbody>")
            for r in scr["results"]:
                m = r["metrics"]
                H.append(f"<tr><td><b>{r['symbol']}</b></td><td>{'✓' if r['passed'] else '✗'}</td><td>{r['score']}</td><td>{r['setup']}</td>"
                         f"<td>{m.get('rsi14')}</td><td>{m.get('atr_pct')}</td><td>{m.get('rel_volume')}</td><td>{m.get('ret63_pct')}</td>"
                         f"<td>{', '.join(r['failed'])}</td></tr>")
            H.append("</tbody></table></div></details>")
        for s in [x for x in sessions if x["day"] == day]:
            tr = parse_transcript(run / s["transcript"]) if (run / s["transcript"]).exists() else {"agents": [], "guard": [], "calls": []}
            who = "You (harness)" if s["turn"] == 2 else "You"
            H.append(f"<div class='turn'><div class='bubble user'><span>{who}</span>{md(s['prompt'].split('[HISTORICAL REPLAY')[0] if s['turn'] == 2 else s['prompt'].split(chr(10) + chr(10), 1)[-1])}</div>")
            if tr["agents"]:
                H.append("<ol class='agents'>")
                for a in tr["agents"]:
                    rep = a.get("report") or "(no report)"
                    H.append(f"<li><details><summary><b>{html.escape(a['agent'])}</b> — {html.escape(a['description'])}</summary>"
                             f"<div class='brief'><h5>Brief from the desk lead</h5>{md(a['brief'][:4000])}</div>"
                             f"<div class='report'><h5>Agent report</h5>{md(rep[:12000])}</div></details></li>")
                H.append("</ol>")
            orders = [c for c in tr["calls"] if any(k in c["tool"] for k in ("place_stock_order", "cancel_order", "close_position", "replace_order", "create_alert"))]
            for c in orders:
                H.append(f"<div class='call'><code>{html.escape(c['tool'].replace('mcp__', ''))}</code> {html.escape(json.dumps(c['input'])[:400])}</div>")
            for g in tr["guard"]:
                H.append(f"<div class='guard'>{html.escape(g[:500])}</div>")
            H.append(f"<div class='bubble desk'><span>Desk lead · {s.get('num_turns')} turns · {s.get('seconds', 0) // 60} min</span>{md(s.get('result') or '(no final message)')}</div></div>")
        H.append("</article>")
    H.append("</section>")

    # journal review
    revs = sorted((ws / "reports").glob("review-*.md")) if (ws / "reports").exists() else []
    if revs:
        H.append(f"<section id='review'><h2>Journal review ({revs[-1].stem})</h2><div class='card'>{md(revs[-1].read_text())}</div></section>")

    H.append(f"""<section id='method'><h2>How this replay works</h2><ul>
<li><b>Data:</b> real daily OHLCV for 88 US stocks (stocknet dataset, Yahoo Finance, 2012-09 → 2017-09); SPY as a total-return index rebuilt from zipline's daily SPY returns (closes only); VIX from the datahub finance-vix dataset.</li>
<li><b>No look-ahead in the tools:</b> every connector call returns data only up to the replay date. Web search and fetch were disabled for the desk. The model may still carry general knowledge of 2017 markets; the desk's scripts (screen, backtest, Kelly, risk) make the numeric decisions.</li>
<li><b>Fills:</b> orders placed after a session's close become eligible the next trading day and fill against real bars: limit at the open or the limit, stop-limit on trigger, bracket legs good-til-cancelled, stop assumed first when a bar touches both, gaps fill at the open. No commissions; no partial fills.</li>
<li><b>The human:</b> the harness supplied the money rules and replied "{html.escape('Yes, place it exactly as shown.')}" whenever the desk showed an order card. The risk manager's approval ticket and the order-guard hook still had to pass for any order to reach the broker.</li>
<li><b>No news:</b> the replay has no historical news feed, so the news desk could not check headlines or confirm earnings dates.</li>
</ul></section>""")
    notes = run / "notes.md"
    if notes.exists():
        H.append(f"<section id='notes'><h2>Harness notes</h2>{md(notes.read_text())}</section>")
    H.append("</main></body></html>")
    outdir = run / "report"
    outdir.mkdir(exist_ok=True)
    out = outdir / "index.html"
    out.write_text("\n".join(H))
    return out


CSS = """
:root{--bg:#f6f7f5;--surface:#fff;--surface2:#eef1ee;--text:#17211e;--muted:#5b6863;--line:#d8dfdb;--accent:#0f766e;--up:#15803d;--down:#b42318;--warn:#8a5a00;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0d1412;--surface:#141d1a;--surface2:#1b2623;--text:#e5eeeb;--muted:#9aaaa4;--line:#27342f;--accent:#2dd4bf;--up:#4ade80;--down:#f87171;--warn:#f3c969}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 var(--sans)}
.wrap{max-width:1120px;margin:0 auto;padding:0 18px 60px}
.hero{padding:40px 0 10px}.eyebrow{font:600 12px var(--mono);letter-spacing:.05em;color:var(--accent)}
h1{font-size:clamp(28px,4.5vw,42px);line-height:1.1;margin:10px 0 12px;letter-spacing:-.01em}
.lede{color:var(--muted);max-width:80ch;margin:0}
h2{font-size:24px;margin:34px 0 12px}h3{margin:0 0 8px}h5{margin:10px 0 6px;font-size:12px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
section{border-top:1px solid var(--line);margin-top:26px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin:22px 0 4px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.kpi span{display:block;color:var(--muted);font-size:13px}.kpi b{display:block;font:700 22px var(--mono);margin:2px 0}.kpi small{color:var(--muted)}
.chart{width:100%;height:auto;background:var(--surface);border:1px solid var(--line);border-radius:12px;margin-top:8px}
.chart .grid{stroke:var(--line);stroke-width:1}.chart .ax{fill:var(--muted);font:11px var(--mono)}.chart .mk{font:600 11px var(--sans)}
.legend{color:var(--muted);font-size:13px}.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin:0 6px 0 12px}.sw.a{background:var(--accent)}.sw.m{border-top:2px dashed var(--muted)}
.tw{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--surface)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{background:var(--surface2);font-weight:600}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}.muted{color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}
.session{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin:16px 0}
.day{font:700 20px var(--mono)}.chips{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}
.chip{font-size:12.5px;padding:3px 9px;border-radius:999px;background:var(--surface2);border:1px solid var(--line)}
.chip.ok{border-color:var(--up)}.chip.bad{border-color:var(--down)}
.bubble{border-radius:12px;padding:10px 14px;margin:10px 0;border:1px solid var(--line)}
.bubble>span{display:block;font:600 12px var(--mono);color:var(--muted);margin-bottom:4px}
.bubble.user{background:var(--surface2)}.bubble.desk{background:var(--surface);border-left:4px solid var(--accent)}
ol.agents{margin:6px 0;padding-left:22px}ol.agents li{margin:4px 0}
details summary{cursor:pointer}details[open]>summary{margin-bottom:6px}
.brief,.report{border:1px solid var(--line);border-radius:10px;padding:8px 12px;margin:6px 0;background:var(--bg);overflow-wrap:anywhere}
.call{font:12.5px var(--mono);background:var(--surface2);border-radius:8px;padding:6px 10px;margin:6px 0;overflow-wrap:anywhere}
.guard{font:12.5px var(--mono);border:1px solid var(--warn);color:var(--warn);border-radius:8px;padding:6px 10px;margin:6px 0}
pre.code{background:var(--surface2);padding:10px 12px;border-radius:8px;overflow-x:auto;font:12.5px/1.5 var(--mono)}
code{font-family:var(--mono);font-size:.92em}blockquote{border-left:3px solid var(--line);margin:8px 0;padding-left:10px;color:var(--muted)}
p{margin:6px 0}ul,ol{margin:6px 0 6px 0;padding-left:22px}
"""


def shots(html_path: Path, outdir: Path) -> list[Path]:
    js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch();
  const p = await b.newPage({{ viewport: {{ width: 1240, height: 900 }}, deviceScaleFactor: 2 }});
  await p.goto('file://{html_path}');
  const out = [];
  async function el(sel, name) {{ const h = await p.$(sel); if (h) {{ await h.screenshot({{ path: '{outdir}/' + name }}); out.push(name); }} }}
  await el('#top', '01-header.png');
  await el('#kpis', '02-kpis-equity.png');
  await el('#trades', '03-trades.png');
  const sessions = await p.$$('article.session');
  for (let i = 0; i < sessions.length; i++) {{
    await sessions[i].evaluate(e => e.querySelectorAll('details').forEach((d, k) => {{ if (k < 2) d.open = true; }}));
    await sessions[i].screenshot({{ path: '{outdir}/' + String(10 + i) + '-session.png' }}); out.push(String(10 + i) + '-session.png');
  }}
  await el('#review', '90-review.png');
  console.log(out.join('\\n'));
  await b.close();
}})();
"""
    jsf = outdir / "shots.js"
    jsf.write_text(js)
    r = subprocess.run(["node", str(jsf)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:], file=sys.stderr)
    return [outdir / n for n in r.stdout.split()]


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--shots", action="store_true")
    a = ap.parse_args()
    run = HERE / "runs" / a.run
    os.environ["SIM_STATE"] = str(run / "state.json")
    os.environ["SIM_DATA"] = str(HERE / "data")
    out = build(run)
    print(out)
    if a.shots:
        for s in shots(out, out.parent):
            print(s)


if __name__ == "__main__":
    main()
