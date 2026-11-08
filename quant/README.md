# quant — multi-asset research, backtesting and PAPER trading (US + UK)

A Python package for quantitative research on US and UK markets: a swappable data layer with a local
Parquet/DuckDB cache, point-in-time universes, fundamental / technical / factor / macro / statistical
analysis modules, an event-driven multi-currency backtester, portfolio construction, risk analytics and a
simulated paper broker.

> **Research and paper trading only.** There is no live broker in this version. The live-trading gate
> (`quant.config.live_gate`) requires a config flag, an environment variable *and* a typed confirmation, and
> then still raises `NotImplementedError`. Backtests describe the past under stated assumptions; nothing
> here is a claim that any strategy is profitable, or that the code is free of errors. Every report ends
> with a *Caveats* section that lists the assumptions, data-quality flags, survivorship coverage and the
> number of variants tried.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,yfinance]"        # add alpaca / polygon / tiingo / ndl / fred extras as needed
quant version
quant config                             # effective settings, secrets masked
quant data instruments --market XLON     # the seed catalogue
quant data fetch AAPL VOD.L --start 2020-01-01   # daily bars into the local Parquet/DuckDB cache
quant data show VOD.L --tail 5           # pence already converted to pounds
quant data validate AAPL VOD.L --issues  # gaps, stale closes, outliers, GBX/GBP jumps, splits, dividends
quant data macro DGS10                   # FRED (no key needed); quant data macro IUDBEDR for Bank Rate (BoE)
quant data lineage                       # what was fetched, from where, how many rows, content hash
quant universe build sp500               # constituents + reconstructed history (quality flag reported)
quant universe show ftse100 --as-of 2024-06-28
quant screen ftse100 --start 2023-01-01 --html screen.html   # liquidity / vol / data-quality ranking, not a forecast
quant backtest run trend_following --instruments AAPL,JPM,VOD.L --start 2020-01-01 --base GBP \
    --param fast=50 --param slow=200 --sensitivity "fast=20,50;slow=100,200" --walk-forward 252,63 \
    --bootstrap --html tearsheet.html      # add --inline-js for a fully offline report
quant analyze indicators VOD.L           # latest indicators + trend / volatility regime labels
quant analyze regime SPX --vol-index VIX --markov
quant analyze cointegration SHEL.L BP.L  # Engle-Granger, Johansen, hedge ratio, half-life, spread z
quant analyze garch UKX --model GJR      # GARCH-family fit and volatility forecast
quant fundamentals fetch AAPL MSFT       # statements + earnings dates (QUANT_SEC_USER_AGENT enables EDGAR)
quant fundamentals ratios AAPL --as-of 2024-06-28   # only what had been filed by that date
quant fundamentals dcf AAPL --growth 0.06 --terminal 0.025 --beta 1.2
quant fundamentals scores sp500          # percentile ranks across a stored universe (descriptive)
quant factors build ftse_factors --universe ftse100 --start 2018-01-01 --sector-neutral
quant factors exposures runs/trend --factors ftse_factors --rf 0.04   # a backtest --out dir, or an instrument
quant factors kenfrench --file ff5_daily # US Fama-French factors (network)
quant macro catalog --region uk          # catalogued FRED / BoE / ONS series (unverified ids are flagged)
quant macro series us_cpi --start 2015-01-01
quant macro curve --region us            # tenor snapshot, 10y-2y slope, inversion flags
quant macro inflation --region uk        # YoY / 3m annualised / regime labels
quant macro dashboard --region us --html board.html   # cross-asset regime board from cached inputs
quant optimize weights -i AAPL,MSFT,JPM --method hrp --start 2020-01-01 --html weights.html
quant optimize weights -i AAPL,MSFT,JPM --method bl --view AAPL:0.10@0.5 --market-weights AAPL=0.5,MSFT=0.3,JPM=0.2
quant optimize frontier -i AAPL,MSFT,JPM --points 10      # long-only frontier from shrunk historical means
quant optimize size --win-rate 0.55 --payoff 1.8 --trades 60 --entry 100 --stop 95 --capital 100000
quant risk var -i AAPL,MSFT,JPM --weights AAPL=0.5,MSFT=0.3,JPM=0.2 --alpha 0.99   # VaR family + Kupiec
quant risk stress -i AAPL,MSFT,JPM --start 2007-01-01 --shock crash:AAPL=-0.3,MSFT=-0.2
quant risk drawdown AAPL --start 2015-01-01   # or --backtest runs/trend for a saved equity curve
quant risk correlation -i AAPL,MSFT,JPM --window 63
quant risk report -i AAPL,MSFT,JPM --html risk.html
quant paper init demo --cash 1000000 --as-of 2024-06-18          # sim broker, multi-currency, DuckDB ledger
quant paper run demo --strategy fixed_weights -i AAPL,MSFT --weights AAPL=0.05,MSFT=0.04 --html blotter.html
quant paper submit demo -i AAPL --side buy --qty 10 --type limit --limit 150 --stop-loss 140 --take-profit 180
quant paper status demo && quant paper reconcile demo
quant paper live                                                 # prints why live trading is refused
quant data adapters                                              # configured, keyed-but-unset, and licensed stubs
quant data fetch AAPL --vendor stooq                             # key-free fallback for daily bars
quant data filings AAPL --forms 10-K,10-Q --since 2023-01-01     # EDGAR submissions (QUANT_SEC_USER_AGENT)
quant data filings VOD --vendor companies_house                  # UK filing history (QUANT_COMPANIES_HOUSE_API_KEY)
```

Configuration is layered: package defaults (`quant/config/default.yaml`) → your YAML (`quant --config my.yaml`)
→ `.env` → environment variables (`QUANT_TRADING__BASE_CURRENCY=GBP`, `QUANT_FRED_API_KEY=…`). API keys are
only ever read from the environment or `.env`, never from YAML.

## Tests

```bash
pytest tests/quant -q          # no network: sockets, requests and yfinance are blocked by conftest.py
pytest -m live tests/quant     # opt-in smoke tests against real endpoints (never run in CI)
ruff check quant tests/quant
```

## Status

Built in tested stages; see the repository history. Each stage's status note states what passed, what was
skipped and known gaps. Completed: **S0 scaffold**, **S1 data core** (adapter contract, yfinance adapter,
Parquet/DuckDB store with idempotent appends, lineage, incremental `DataService`), **S2 data quality**
(validation report with grades, corporate actions and as-of adjustment, FX with forward-fill-only alignment,
FRED and Bank of England macro series), **S3 backtest engine** (event-driven, multi-currency, next-open fills,
itemised costs incl. UK/US transaction taxes, look-ahead guards), **S4 universes** (Wikipedia constituents with
shape checks, point-in-time membership with snapshot accumulation and S&P change-table reconstruction,
coverage, evidence-based screen, HTML report builder), **S5 robustness** (walk-forward, parameter sensitivity,
block-bootstrap Sharpe interval, deflated Sharpe ratio fed by a per-process variant counter, tearsheet with
light/dark charts and table views, cross-sectional momentum, `quant backtest run`). **S6 technical + stats** (Wilder RSI/ATR, MACD, ADX, stochastic, Bollinger/Keltner, OBV/VWAP/MFI, pivot
support/resistance, volatility, trend, rule-based and Markov-switching regimes; ADF/KPSS/PP/variance-ratio,
Engle-Granger/Johansen, GARCH, PCA/absorption ratio, OLS with HAC and diagnostics; cointegration pairs strategy;
`quant analyze`), **S7 fundamentals** (point-in-time statements with period end *and* filing date from yfinance
(estimated lag, flagged) or SEC EDGAR companyfacts (true dates); ratios on a shared basis, FCFF DCF with WACC and
a sensitivity grid, comparables, cross-sectional quality/value/growth scores, earnings surprise and SUE;
`quant fundamentals`), **S8 factors** (12-1 momentum, low-vol, size, point-in-time value and quality
characteristics; rank-normalisation with optional sector neutrality; dollar-neutral quantile long-short factor
returns with drift between monthly rebalances; HAC exposures with injectable risk-free rate; exact arithmetic
attribution; Ken French loader; `quant factors`), **S9 macro** (ONS time-series adapter; a catalogue of
US and UK series with per-series `verified` flags, several UK BoE/ONS ids still unverified against the live
APIs; yield-curve slope, curvature, inversion flags and tenor snapshots; inflation transforms and a
rule-based regime label; a growth x inflation quadrant and cross-asset board that lists missing inputs
rather than imputing them; `quant macro`), **S10 portfolio** (sample / EWMA / Ledoit-Wolf covariance with
PSD repair; SLSQP mean-variance with box, budget, gross and group constraints and reported binding
constraints; min-variance, max-Sharpe, target-return, target-risk and the efficient frontier; risk parity in
Spinu's convex form plus a bounded fallback; Black-Litterman with absolute and relative views; HRP; ex-ante
and realised vol targeting with lagged leverage; fractional-Kelly sizing with Wilson shrinkage and ordered
caps; a vectorised periodic rebalance pinned to the engine by an equivalence test; `FixedWeights` strategy;
`quant optimize`), **S11 risk** (historical, normal, Student-t, Cornish-Fisher, GARCH filtered-historical and
Monte Carlo VaR/CVaR with a Kupiec backtest of out-of-sample rolling VaR; dated stress windows from the GFC to
the 2022 gilt crisis with coverage reported, hypothetical and beta shocks, factor scenarios from estimated
betas; calm-versus-stress correlation breakdown, diversification and absorption ratios; drawdown episodes,
ulcer/pain indices, CDaR; an HTML risk report; `quant risk`), **S12 paper trading** (Broker protocol; a
multi-currency SimBroker that re-uses the engine's fill rules and cost model and is pinned to the engine by a
fill-for-fill equivalence test; a DuckDB ledger whose cash and positions are re-derivable from the event log;
a pre-trade gate ported from the plugin's rule list; a daily runner that replays missed sessions, lets the
strategy decide on data up to the close, gates and submits orders and reconciles; an HTML blotter; an Alpaca
adapter that refuses `paper=False`; `quant paper`), **S13 adapters** (Stooq, Alpaca market data, Polygon
aggregates + reference dividends/splits, Tiingo prices with embedded actions, Nasdaq Data Link datasets as
macro series, Companies House filing history, EDGAR submissions filings index; keyed vendors register only
when their credentials are set; Bloomberg / Refinitiv / FactSet / Capital IQ stubs that declare their
intended capabilities, advertise none until wired, and raise with concrete steps; a crypto helper module
for 24/7 series; one contract test over every adapter; `quant data adapters|filings`). Next: S14 docs and
end-to-end demo.

## Module map

| Package | Purpose |
|---|---|
| `quant.config` | settings, date-effective fee tables, instrument seed catalogue, live-trading gate |
| `quant.core` | `Instrument` and enums, exchange calendars (XNYS, XLON, FX, CRYPTO, US futures), the `now()` seam, errors |
| `quant.data` | adapters (yfinance, FRED, BoE, ONS, EDGAR, …; licensed stubs), Parquet/DuckDB store, validation, FX, lineage |
| `quant.universe` | index constituents, point-in-time membership, liquidity/volatility/data-quality screens |
| `quant.fundamental` | PIT statements, ratios, DCF, comparables, quality/value/growth scores, earnings surprise |
| `quant.technical` | trend, momentum, volatility, volume indicators; support/resistance; regimes |
| `quant.factors` | factor construction, long-short returns, exposures and attribution |
| `quant.macro` | rates, inflation, yield curve, cross-asset regime board |
| `quant.stats` | stationarity, cointegration, GARCH, PCA, regression diagnostics |
| `quant.portfolio` | mean-variance, risk parity, Black-Litterman, HRP, vol targeting, sizing |
| `quant.risk` | VaR/CVaR, stress tests, scenarios, correlation breakdown, drawdowns |
| `quant.backtest` | event-driven multi-currency engine, orders/fills, costs, metrics, robustness analysis |
| `quant.paper` | broker protocol, SimBroker, Alpaca paper adapter, pre-trade gate, ledger, daily runner |
| `quant.reports` | self-contained HTML reports |
| `quant.cli` | the `quant` command |

## Known limitations (kept current as stages land)

- Daily bars only. Intraday is supported by the storage schema but not by the backtester.
- Free data sources give **approximate** point-in-time universes. FTSE 100/250 history is `current_only`
  until snapshots accumulate; every backtest reports its survivorship coverage.
- Transaction taxes and levies are modelled from a date-effective table (`quant/config/fees.yaml`); entries
  marked `verified: false` have not yet been checked against the primary source.
