# Connectors

The desk needs market data, a broker and (ideally) alerts. It is built for two connectors, and degrades gracefully when one is missing.

| Need | Preferred | Fallbacks |
|---|---|---|
| Watchlists | TradingView connector | user types the symbols |
| Daily price history | `fetch_bars.py` (Alpaca data API, exact) · TradingView OHLCV | Alpaca `get_stock_bars` · TradingView "Export chart data" CSV |
| Indicator snapshots for big watchlists | TradingView screener / quote tools | computed from bars by `screen.py` |
| News, earnings dates | TradingView news/calendar · Alpaca `get_news` | WebSearch / WebFetch (company IR pages) |
| Price alerts to the phone | TradingView alert tool | manual alert steps for the user |
| Account, positions, orders | Alpaca connector (paper) | none — no trading without it |

## TradingView

Add it in Claude's settings under connectors (the exact menu name changes between app versions; look for *Connectors* → add/browse). Notes:

- TradingView's MCP access may require a paid TradingView plan; check the connector's own setup page.
- Tool names differ between connectors and versions. Never hard-code them: agents find them with ToolSearch (`tradingview`, `watchlist`, `chart`, `alert`) and read each tool's input schema.
- Symbols usually need an exchange prefix (`NASDAQ:NVDA`, `NYSE:JPM`, `AMEX:SPY`, `CBOE:VIX`). Files in the workspace use the bare ticker (`NVDA.csv`).
- Some TradingView connectors are read-only. If there is no alert-creation tool, the alerts agent returns manual steps.
- Data returned by an MCP tool has to be written to a file by the model. For long histories (1,000+ bars) prefer `fetch_bars.py`, which downloads directly and cannot mistype a number. Always check the bar count and date range that `save_bars.py` prints against what the tool returned.

## Alpaca (official `alpaca-mcp-server`)

1. Create a free Alpaca account and generate **paper trading** API keys in the Alpaca dashboard.
2. Add the server. For Claude Code (requires `uv`):

   ```bash
   claude mcp add alpaca --scope user \
     -e ALPACA_API_KEY=<paper key id> -e ALPACA_SECRET_KEY=<paper secret> -e ALPACA_PAPER_TRADE=true \
     -- uvx alpaca-mcp-server
   ```

   For Claude Desktop, add the same command, args and env to `claude_desktop_config.json` under `mcpServers.alpaca`. Check the server's README for the current instructions — the v2 server renamed its tools and settings.
3. Name the server **`alpaca`**. The agents' tool deny-lists assume the tool prefix `mcp__alpaca__`; the order-guard hook matches any server name containing `alpaca`.
4. `ALPACA_PAPER_TRADE` defaults to `true`. Keep it that way. Live trading needs `ALPACA_PAPER_TRADE=false` there **and** a hand edit of `profile.json` — two separate, deliberate human steps.
5. Optional `ALPACA_TOOLSETS` (e.g. `account,trading,stock-data,news,assets`) trims the tool list.

Tools the desk uses (v2 names): `get_account_info`, `get_all_positions`, `get_open_position`, `get_orders`, `get_order_by_id`, `place_stock_order`, `replace_order_by_id`, `cancel_order_by_id`, `close_position`, `get_stock_bars`, `get_stock_latest_quote`, `get_news`, `get_clock`, `get_calendar`, `get_account_activities`.

### Optional: keys in the shell for `fetch_bars.py`

`fetch_bars.py` reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` (or `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY`) from the environment Claude Code runs in — e.g. exported in your shell profile. The free data plan serves the IEX feed. Never paste keys into the chat.

## Permissions

Leave Alpaca's order tools (`place_stock_order`, `replace_order_by_id`, `close_position`, `close_all_positions`) **out** of any auto-allow list, so Claude Code asks you before each one runs — a second human check on top of the in-chat confirmation and the order-guard hook. Read-only tools (`get_*`) can be allowed to cut down on prompts.
