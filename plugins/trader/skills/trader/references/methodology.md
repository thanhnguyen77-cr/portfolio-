# Methodology

All calculations live in `scripts/`. This page explains what they do and why, so the desk can explain it to the user in plain words.

## Market regime (`regime.py`)
Points: SPY and QQQ each get +1 for close > 50-day average, +1 for close > 200-day, +1 for 50-day > 200-day. VIX < 20 adds 1, VIX > 30 subtracts 2. SPY more than 10% below its 52-week high subtracts 1. With all three inputs (max 7): ≥ 6 risk_on (full size), ≥ 3 caution (half size), else risk_off (no new longs).

## Screener (`screen.py`)
Hard filters (all must pass; missing data counts as a fail, not a pass):
- price ≥ `min_price`; average daily dollar volume ≥ `min_avg_dollar_volume`;
- close > SMA50 and SMA50 > SMA200 (an established uptrend);
- RSI14 within `[rsi_min, rsi_max]` (default 45–70: momentum, not overbought);
- today's volume ≥ `min_relative_volume` × 20-day average;
- ATR14 as % of price ≤ `max_atr_pct` (avoid names where a normal day blows through any sane stop).

Score 0–100: trend 30 (distance above SMA50 with a penalty when over-extended, SMA50 vs SMA200, SMA50 slope), momentum 25 (3-month return), RSI 15 (best near 60), volume 15, proximity to the 52-week high 15; minus a volatility penalty above 5% ATR.

Setup label: `breakout` (within 2% of the prior 20-day high on ≥ 1.2× volume), `pullback` (within 3% of the 20-day EMA with RSI ≤ 58), else `trend_continuation`.

## The four setups (`backtest.py`)
Signals are evaluated on a bar's close using only data up to that bar; entries fill at the next bar's open plus slippage.
- **breakout** — uptrend (close > SMA50 > SMA200), close above the highest high of the prior `lookback` (20) bars, volume ≥ `min_relvol` × the prior 20-day average.
- **pullback** — close > SMA200, SMA50 > SMA200, the bar's low touches the 20-day EMA and it closes above it, RSI14 ≤ 60.
- **trend_continuation** — uptrend and RSI14 crosses up through 50.
- **mean_reversion** — close > SMA200 and RSI2 < 10.

## Backtest rules
- Stop = entry − `stop_atr` × ATR14 at the signal bar (or `stop_pct`%); target = entry + `rr` × that distance; time stop after `max_hold` bars.
- If a bar touches both stop and target, the **stop** is assumed first. A gap through the stop fills at the open (worse than the stop). One position at a time per symbol.
- Costs per side: slippage bps on the price, fee bps on notional, commission per order and per share. Defaults (5 bps slippage, 0.5 bps fees, $0 commission) suit liquid US stocks on a commission-free broker; raise slippage for thin names.
- **Out-of-sample**: the last 30% of each symbol's history is reported separately. A rule that only worked early is `FAIL_OOS`.
- **Sensitivity**: the same rule with stop and target each at 75%/100%/125% (9 runs). If fewer than half are profitable the result is `FRAGILE`.
- **Verdicts**: `FAIL` (expectancy ≤ 0 or profit factor < 1), `FAIL_OOS`, `MARGINAL` (expectancy < 0.1 R), `FRAGILE`, `INSUFFICIENT_TRADES` (< 30 trades — add peers), `PASS`.
- Known limits: no taxes, partial fills, halts, borrow, or survivorship bias (today's watchlist is made of survivors). Treat results as an upper bound.

## Position sizing (`kelly.py`)
With a hard stop, each trade loses ~1R or wins ~b·R. Kelly's optimal fraction of capital **to risk** is `f* = p − (1 − p) / b`.
1. `p` is shrunk to the 1-sigma Wilson lower bound for the number of trades, so small samples count for less.
2. `f` = max(0, f*) × `kelly_fraction` (¼ by default — full Kelly is far too volatile for estimated edges) × regime multiplier.
3. Dollars at risk = min(f × capital, `max_risk_per_trade_pct` × capital). Risk per share includes round-trip costs.
4. Shares are then capped by `max_position_value`, cash above `cash_reserve`, and `max_total_exposure_pct`, and rounded down.
Usually the 1% risk cap binds before Kelly does. That is intended: Kelly mainly answers "is there enough edge to take any risk at all?".

## Risk check (`risk_check.py`) and the order guard (`order_guard.py`)
`risk_check.py` runs ~20 checks (see the risk-manager agent) and, on APPROVE with `--approve`, writes a single-use ticket valid for `approval_ttl_minutes`. `order_guard.py` runs as a Claude Code PreToolUse hook on Alpaca tools: read-only calls, cancels and position closes pass; anything that places or changes an order needs a matching unexpired, unused ticket (same symbol and side, qty ≤ approved, limit no worse than approved, protective stop attached for entries). Options and crypto orders are always blocked.
