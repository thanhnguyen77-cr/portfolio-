# Historical replay harness

This folder runs the **real** Trader plugin — all 11 agents, the desk-lead skill, the scripts and the order-guard hook — in headless Claude Code. The run happens against **real historical market data**, with a simulated paper broker. Nothing about the desk is mocked. Only its two connectors are swapped for replay versions, which serve data up to the session date and nothing later.

```
python3 sim/fetch_data.py                                    # real data -> sim/data/
python3 sim/run_sim.py init   --run replay2017 --start 2017-05-05
python3 sim/run_sim.py day    --run replay2017 --date 2017-05-05
python3 sim/run_sim.py day    --run replay2017 --date 2017-06-02
python3 sim/run_sim.py review --run replay2017 --date 2017-09-01
python3 sim/report.py --run replay2017 --shots               # HTML report + screenshots
python3 sim/watch.py sim/runs/replay2017                     # optional: live feed of decisions
```

## Pieces

| File | Role |
|---|---|
| `fetch_data.py` | Downloads real daily data: 88 US stocks 2012-09 to 2017-09 (stocknet / Yahoo Finance), SPY as a total-return index rebuilt from zipline's daily SPY returns, and VIX from datahub's finance-vix. |
| `engine.py` | Replay clock, account, Alpaca-style orders (limit, stop, stop-limit, bracket legs), fills against real bars, TradingView-style alerts. |
| `alpaca_server.py` | Simulated Alpaca MCP connector: account, positions, orders, bars, clock, calendar and activities. Uses the same tool names as the official server. |
| `tradingview_server.py` | Simulated TradingView MCP connector: watchlists, indicator snapshots, OHLCV, chart-data export to CSV, and alerts. |
| `mcp_stdio.py` | A dependency-free MCP stdio server, around 100 lines. |
| `run_sim.py` | The driver. It advances the replay, starts a fresh headless `claude -p` session per trading day with `--plugin-dir plugins/trader`, and plays the human. |
| `report.py` | Builds an HTML report from broker ground truth, workspace files and transcripts, and takes screenshots with Playwright. |

## What the harness does as "the user"

- On day one it types `install skill` and gives the money rules: $10,000 budget, $2,000 cash aside, $2,500 max per stock.
- Each session it asks for the day's trading loop on the watchlist "Swing 2017".
- If the desk shows an order card, it replies **"Yes, place it exactly as shown."** It never calls a broker tool itself. The risk manager's approval ticket and the order-guard hook still have to pass.

## Guarding against look-ahead

- Every connector call returns data only up to the replay date. Orders become eligible the next trading day and fill on later bars.
- `WebSearch` and `WebFetch` are disabled for the desk, since the live web would leak the future. As a result there is no news feed.
- The model may still carry general knowledge of markets from its training. The numeric decisions come from the desk's scripts: screen, backtest, Kelly sizing and risk check.

## Fill model

- A limit buy fills at the open if the open is at or below the limit, otherwise at the limit if the day's low reaches it.
- A stop-limit buy triggers when the high reaches the stop.
- Bracket legs are good-til-cancelled. If one bar touches both the stop and the target, the stop counts first. A gap through a leg fills at the open.
- No commissions and no partial fills.
