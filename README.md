# Trader Skill: an 11-agent trading desk for Claude

The Trader Skill turns Claude into a small trading desk for US stocks. It has eleven specialised agents. They read your TradingView charts and watchlists, plan and backtest trades, size positions with a capped Kelly Criterion, watch the news, set price alerts, and place **paper** trades on Alpaca. A risk manager can veto any order, and Claude Code itself enforces that veto. Every calculation runs in plain, tested Python, never in the model's head.

> **Not financial advice.** This is a decision-support and paper-trading tool. Backtests describe the past, stops can gap, and you are responsible for every order. The desk starts in paper mode and cannot switch itself to live trading.

## Install

**Option A: Claude Code plugin (recommended).** This installs the skill, all 11 agents and the order-guard hook in one go:

```
/plugin install trader --marketplace thanhnguyen77-cr/portfolio-
```

On older Claude Code versions:

```
/plugin marketplace add thanhnguyen77-cr/portfolio-
/plugin install trader@portfolio-trader
```

After that, choose **Opus 5.5** with `/model` and type **"install skill"**. The desk creates its workspace, asks for your money rules and checks your connectors.

**Option B: standalone skill.** Run `python3 tools/build_skill_zip.py`, then either:
- copy `dist/trader/` to `~/.claude/skills/trader/` and type **"install skill"** in Claude Code. `install.py` copies the 11 agents to `~/.claude/agents/` and registers the order-guard hook in your project's `.claude/settings.json`; or
- upload `dist/trader-skill.zip` in claude.ai under *Customize → Skills*. Chat has no subagents, so there Claude plays each role in turn by following the agent files. It needs code execution turned on to run the scripts.

## Connect your accounts

| Connector | What the desk uses it for | Setup |
|---|---|---|
| **TradingView** | watchlists, charts and price history, indicator snapshots, news, price alerts | Add it in Claude's connector settings. TradingView may require a paid plan for MCP access. |
| **Alpaca** (paper) | account, positions, orders, and also price history and news | `claude mcp add alpaca --scope user -e ALPACA_API_KEY=… -e ALPACA_SECRET_KEY=… -e ALPACA_PAPER_TRADE=true -- uvx alpaca-mcp-server` |

Full details, fallbacks and permission tips are in [`references/connectors.md`](plugins/trader/skills/trader/references/connectors.md). Name the Alpaca server `alpaca`, and keep its order tools out of your auto-allow list so Claude Code asks you before every order.

## How a trade flows

| Your step | What happens | Agent → script |
|---|---|---|
| 1. "install skill" | Creates the workspace, installs agents and hook if standalone, checks setup | desk lead → `install.py` |
| 2. Add connectors | TradingView and Alpaca (paper) are checked with read-only calls | `trader-onboarding` |
| 3. Budget, cash to keep aside, max per stock | Saved as validated money rules. Defaults: 1% risk per trade, ¼ Kelly | `trader-onboarding` → `trader_profile.py` |
| — | The market's trend decides the size: risk_on, caution (half size) or risk_off (no new longs) | `trader-market-regime` → `regime.py` |
| 4. Scan the watchlist | Filters on trend (50/200-day), RSI band, volume, volatility and liquidity. Survivors get a 0–100 score | `trader-screener` → `screen.py` |
| 5. Trade plan | Exact entry, stop and target, the reasons for the trade, and what would invalidate it | `trader-analyst` |
| 6. Backtest | The plan's rule on 5+ years of daily bars plus peer stocks, after slippage and fees, with an out-of-sample check and a 9-setting sensitivity grid. Gives an honest verdict | `trader-backtester` → `backtest.py` |
| 7. Size and news | Fractional Kelly using a shrunk win rate, capped by risk per trade, max per stock, cash reserve and exposure. The news desk rates earnings, filings and headlines green, amber or red, and can keep watching | `trader-position-sizer` → `kelly.py`, `trader-news-desk` |
| 8. Alerts | TradingView alerts at the entry and invalidation levels ping your phone. If the connector can't create alerts, you get exact manual steps | `trader-alerts` |
| 9. Paper order | You confirm an order card. The risk manager re-checks every rule against a fresh account snapshot and issues a single-use ticket. The broker then places a bracket order on Alpaca paper | `trader-risk-manager` → `risk_check.py`, `trader-broker` |
| — | Fills are logged, results compared with the backtest, and lessons written up | `trader-journal` → `journal.py` |

"No trade" is a normal and frequent result. Each step has a gate, and the desk stops at the first one that fails.

## Safety design

- **Paper first.** Live trading takes two deliberate human steps: setting `ALPACA_PAPER_TRADE=false` in your connector, and editing `profile.json` by hand. `trader_profile.py` refuses to change either setting.
- **Hard gate on orders.** `order_guard.py` runs as a Claude Code `PreToolUse` hook on Alpaca tools. Read-only calls, cancels and closes pass through. Any other order needs an unexpired, unused approval ticket from `risk_check.py` with the same symbol and side, no more shares, a limit no worse than approved, and a protective stop attached. Options and crypto orders are always blocked.
- **Two human checks per order, plus the gate.** You confirm the order card in chat, Claude Code asks before the order tool runs (keep it off auto-allow), and the hook needs an approval ticket no older than 30 minutes.
- **Least privilege.** Only `trader-broker` may use order tools. The other ten agents have them in `disallowedTools`.
- **No hand arithmetic, no made-up data.** Every number comes from a connector or a script, and agents must say which.

## Limitations

- TradingView connectors differ, so agents find tools by searching rather than by hard-coded names. Some connectors are read-only, and alerts then fall back to manual steps.
- Price history fetched through an MCP tool has to be written to a file by the model. For long histories, `fetch_bars.py` downloads directly from Alpaca's data API when your keys are in the shell environment.
- Backtests ignore taxes, partial fills, halts and survivorship bias. Treat the results as an upper bound.
- Long-only US stocks and ETFs. No shorts, options or crypto.

## Repository layout

```
.claude-plugin/marketplace.json        marketplace "portfolio-trader"
plugins/trader/
  .claude-plugin/plugin.json
  agents/trader-*.md                   the 11 agents
  hooks/hooks.json                     order guard (PreToolUse)
  skills/trader/
    SKILL.md                           desk lead: install, loop, rules
    scripts/                           traderlib, screen, backtest, kelly, risk_check, order_guard,
                                       regime, fetch_bars, save_bars, journal, trader_profile, install
    references/                        connectors, workspace formats, methodology
    templates/                         example profile and plan
tests/test_trader.py                   unit + CLI tests
tools/build_skill_zip.py               builds the standalone skill / zip
```

## Development

```
python3 -m unittest discover -s tests -v     # standard library only
claude plugin validate ./plugins/trader      # plugin manifest, agents, skill
claude plugin validate .                     # marketplace manifest
```
