# Workspace layout and file formats

Everything the desk produces lives in `./trader-workspace/` (or `$TRADER_WORKSPACE`). The folder has its own `.gitignore` so account data never gets committed.

```
trader-workspace/
├── paths.json                 scripts_dir, skill_dir, install mode   (install.py)
├── profile.json               money rules                            (trader_profile.py)
├── data/SYM.csv               daily OHLCV: time,open,high,low,close,volume
├── regime/YYYY-MM-DD.json     market regime                          (regime.py)
├── screens/YYYY-MM-DD.json|md screener output                        (screen.py)
├── plans/SYM-YYYY-MM-DD.json|md  trade plans                         (analyst)
├── backtests/SYM-YYYY-MM-DD.json|md                                  (backtest.py)
├── news/SYM.json              news desk status + log
├── sizing/SYM-YYYY-MM-DD.json position size                          (kelly.py)
├── alerts/alerts.json         alert register
├── orders/                    account snapshots, order requests, broker responses
├── risk/SYM-...json           risk-check results                     (risk_check.py)
├── risk/approvals/*.json      single-use approval tickets            (risk_check.py --approve)
├── journal.csv                trade journal                          (journal.py)
└── reports/review-*.md        performance reviews
```

## profile.json

Created by `trader_profile.py init`; defaults in `scripts/traderlib.py` (`PROFILE_DEFAULTS`); example in `templates/profile.example.json`.

| Key | Default | Meaning |
|---|---|---|
| `budget` | — | total money this desk may use |
| `cash_reserve` | — | cash that always stays uninvested |
| `max_position_value` | — | max $ in one stock |
| `max_risk_per_trade_pct` | 1.0 | max % of capital lost if a stop is hit |
| `kelly_fraction` | 0.25 | fraction of full Kelly used |
| `max_open_positions` | 5 | |
| `max_total_exposure_pct` | 80 | % of budget invested at once |
| `max_daily_loss_pct` | 3.0 | no new entries after this loss today |
| `min_reward_risk` | 2.0 | target distance ÷ stop distance |
| `min_price`, `min_avg_dollar_volume` | 5, 20M | tradability filters |
| `allow_shorts`, `allow_market_orders` | false, false | |
| `require_backtest`, `backtest_min_trades` | true, 30 | |
| `approval_ttl_minutes` | 30 | how long a risk approval stays valid |
| `mode`, `live_trading_enabled` | paper, false | both must be changed by hand for live |
| `blocklist` | [] | symbols never to trade |
| `screen.*` | see defaults | screener thresholds and top-N |
| `fees.*` | 0, 0, 0.5 bps, 5 bps | commission/order, commission/share, fee bps per side, slippage bps per side |
| `guard.require_bracket` | true | order guard demands a protective stop on entries |
| `guard.max_limit_slippage_pct` | 0.5 | how far a submitted limit may differ from the approved one |

## Plan (analyst)

```json
{"symbol": "NVDA", "setup": "breakout", "as_of": "2026-11-06",
 "entry_type": "stop_limit", "entry": 152.30, "entry_limit": 152.90,
 "stop": 146.10, "target": 164.70, "atr14": 3.05, "stop_atr": 2.03, "rr": 2.0,
 "max_hold": 20, "lookback": 20,
 "invalidation": ["daily close below 147.50 before entry"], "valid_until": "2026-11-13"}
```
`setup` ∈ `breakout | pullback | trend_continuation | mean_reversion` — the only rules `backtest.py` can replay.

## Account snapshot (sizer, risk manager)

```json
{"equity": 10250.0, "last_equity": 10180.0, "cash": 6100.0, "buying_power": 12200.0,
 "positions": [{"symbol": "AAPL", "qty": 5, "market_value": 950.0}],
 "open_orders": [{"symbol": "MSFT", "side": "buy", "qty": 3, "limit_price": 410.0}],
 "source": "alpaca get_account_info + get_all_positions + get_orders", "fetched_at": "..."}
```

## Order request (risk manager → broker)

```json
{"symbol": "NVDA", "side": "buy", "qty": 9, "order_type": "stop_limit",
 "stop_price": 152.30, "limit_price": 152.90, "stop_loss": 146.10, "take_profit": 164.70,
 "time_in_force": "day"}
```
Exit: `{"symbol", "side": "sell", "qty", "order_type", "limit_price"}` with `risk_check.py --exit`. Modify: add `"replace_order_id"`.

## News (news desk)

```json
{"symbol": "NVDA", "checked_at": "...", "severity": "green|amber|red",
 "days_to_earnings": 12, "earnings_date": "2026-11-19", "earnings_confirmed": true,
 "flags": [], "items": [{"time": "...", "headline": "...", "source": "...", "impact": "low", "why": "..."}],
 "log": [{"checked_at": "...", "severity": "green", "new_items": 0}]}
```

## Approval ticket (risk_check.py --approve; consumed by order_guard.py)

```json
{"id": "NVDA-20261106T143000-1a2b3c4d", "status": "APPROVED|USED", "kind": "entry|exit|replace",
 "symbol": "NVDA", "side": "buy", "qty": 9, "order_type": "stop_limit",
 "limit_price": 152.9, "stop_price": 152.3, "stop_loss": 146.1, "take_profit": 164.7,
 "mode": "paper", "created_at": "...", "expires_at": "...", "consumed_at": null}
```
Tickets are written only by `risk_check.py`. Never create or edit them by hand.
