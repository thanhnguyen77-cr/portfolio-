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
quant demo --out data/reports/demo       # OFFLINE end-to-end run on SYNTHETIC data; open index.html
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
quant macro verify --region uk           # fetch every catalogued series live; exits 1 if an id fails
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
quant backtest run factor_tilt --universe ftse100 --start 2020-01-01 --base GBP --param top_fraction=0.3
quant backtest run macro_regime_switch -i AAPL,MSFT,SPX,VIX --param benchmark=XNYS_SPX --param vol_index=XNYS_VIX
```

The `examples/` scripts show the same pipeline through the Python API on real data (network needed).

Configuration is layered: package defaults (`quant/config/default.yaml`) → your YAML (`quant --config my.yaml`)
→ `.env` → environment variables (`QUANT_TRADING__BASE_CURRENCY=GBP`, `QUANT_FRED_API_KEY=…`). API keys are
only ever read from the environment or `.env`, never from YAML.

## Tests

```bash
pytest tests/quant -q          # no network: sockets, requests and yfinance are blocked by conftest.py
pytest -m live tests/quant     # opt-in: one small request each to yfinance, FRED and BoE (never run in CI)
ruff check quant tests/quant && ruff format --check quant tests/quant
quant demo --out /tmp/demo     # the offline end-to-end run; exits 1 if any stage fails
```

What the suite does and does not establish: the tests pin behaviour to analytic results (buy-and-hold
returns, mean-variance solutions, VaR closed forms, Kelly fractions), to planted anomalies (gaps, outliers,
splits, denomination changes, look-ahead poison), and to internal consistency (SimBroker fills equal engine
fills; the vectorised rebalance equals the engine; ledger cash and positions re-derive from the event log;
every adapter honours the contract). They do **not** establish that any vendor still serves the formats the
fixtures were recorded from (that is what the opt-in live tests are for), that any strategy earns money, or
that the code is free of errors.

Live checks were run once on 2026-11-08 from the build environment after its network was opened to the
vendor hosts. Outcome: yfinance, FRED and Bank of England requests succeeded and every catalogued FRED and
BoE id resolved (`quant macro verify`), although after a couple of dozen requests the BoE export started
answering an HTML error page with status 200, which the adapter now detects, retries and reports instead of
parsing; the ONS `api.ons.gov.uk` endpoint turned out to have been retired in
November 2024, so the adapter was re-pointed at the ONS website's JSON endpoint and all catalogued ONS ids
now resolve; Wikipedia still serves the S&P 500, FTSE 100 and FTSE 250 constituent tables but has removed
the S&P 500 "selected changes" section and the Nasdaq-100 components list, so S&P history reconstruction and
the Nasdaq-100 universe now need a CSV import; Stooq answered an identified non-browser client with an HTML
page rather than CSV, so that adapter is not usable as-is. SEC EDGAR was not exercised live because it
requires a personal contact address in the User-Agent.

## Status

All fourteen planned stages are implemented, each with tests that passed at commit time (see the repository
history for the per-stage notes). Completed: **S0 scaffold**, **S1 data core** (adapter contract, yfinance
adapter, Parquet/DuckDB store with idempotent appends, lineage, incremental `DataService`), **S2 data
quality** (validation report with grades, corporate actions and as-of adjustment, FX with forward-fill-only
alignment, FRED and Bank of England macro series), **S3 backtest engine** (event-driven, multi-currency,
next-open fills, itemised costs incl. UK/US transaction taxes, look-ahead guards), **S4 universes**
(Wikipedia constituents with shape checks, point-in-time membership with snapshot accumulation and S&P
change-table reconstruction, coverage, evidence-based screen, HTML report builder), **S5 robustness**
(walk-forward, parameter sensitivity, block-bootstrap Sharpe interval, deflated Sharpe ratio fed by a
per-process variant counter, tearsheet with light/dark charts and table views, cross-sectional momentum),
**S6 technical + stats** (indicators, support/resistance, rule-based and Markov regimes; stationarity,
cointegration, GARCH, PCA, OLS diagnostics; pairs strategy), **S7 fundamentals** (point-in-time statements
with period end *and* filing date, ratios on a shared basis, DCF with sensitivity, comparables, scores,
earnings surprise), **S8 factors** (characteristics, rank normalisation, long-short factor returns, HAC
exposures, exact attribution, Ken French loader), **S9 macro** (ONS adapter, series catalogue with
`verified` flags, yield curve, inflation, cross-asset board), **S10 portfolio** (covariance estimators,
mean-variance family, risk parity, Black-Litterman, HRP, vol targeting, Kelly sizing, vectorised rebalance
pinned to the engine), **S11 risk** (VaR/CVaR family with Kupiec backtest, dated stress windows, factor
scenarios, correlation breakdown, drawdown analytics, risk report), **S12 paper trading** (SimBroker pinned
to the engine fill for fill, DuckDB ledger with re-derivable invariants, pre-trade gate, daily runner,
reconciliation, blotter, paper-only Alpaca adapter), **S13 adapters** (Stooq, Alpaca data, Polygon, Tiingo,
Nasdaq Data Link, Companies House, EDGAR filings; licensed stubs; crypto helpers; one contract test over
every adapter), **S14 docs + end-to-end** (`FactorTilt` and `MacroRegimeSwitch` strategies, `quant demo` on
a synthetic mini-dataset exercising every stage offline, examples, a `--help` smoke test over every CLI
command, this status).

Honest summary of where it stands: the research, backtesting and paper-trading pipeline runs end to end
offline and is covered by 400-odd tests; yfinance, FRED, BoE, ONS and the Wikipedia constituent tables have
been exercised live once (see Tests); the Nasdaq-100 list and S&P 500 change history are no longer on
Wikipedia and need an import; some fee-table entries are still marked unverified; and nothing here has been
shown to make money, because that was never the claim.

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
| `quant.crypto` | 24/7 series helpers: 365-day annualisation, alignment onto exchange calendars, caveats |
| `quant.demo` | synthetic mini-dataset and the offline end-to-end `quant demo` |
| `quant.cli` | the `quant` command |

## Known limitations

**Scope**
- Research, backtesting and paper trading only. No live broker exists; the live gate always raises.
- Daily bars only. Intraday is supported by the storage schema but not by the backtester or the broker.
- Long/short with a flat borrow-cost rate; no options; continuous futures come from vendor `=F` series
  without roll modelling, so their long-run returns carry roll gaps.

**Data**
- Free sources give **approximate** point-in-time universes. Wikipedia removed the S&P 500 "selected
  changes" table (observed 2026-11-08), so S&P history can only be reconstructed from a saved copy of the old
  page or an imported CSV of intervals; the Nasdaq-100 components table is gone too, so `ndx100` needs
  `quant universe import`. S&P 500, FTSE 100 and FTSE 250 current constituents still load; all are
  `current_only` until snapshots accumulate. Every backtest reports its survivorship coverage.
- Yahoo closes are split-adjusted but not dividend-adjusted and are stored as such; dividend currencies for
  UK payers that declare in USD are a known soft spot, flagged by validation rather than resolved.
- Vendor response formats are pinned by recorded fixtures and were checked live once (2026-11-08); a later
  vendor change shows up only when you run `pytest -m live tests/quant` and `quant macro verify`.
- Stooq serves an HTML page, not CSV, to identified non-browser clients, so the Stooq adapter currently
  fails with that message; the Bank of England export throttles bursts of requests with an HTML error page
  (the adapter retries three times, then reports it); the Companies House adapter indexes filings but does
  not parse iXBRL accounts;
  Nasdaq Data Link anonymous quotas are small; Alpaca, Polygon, Tiingo, NDL and Companies House have been
  exercised only against recorded fixtures (they need API keys).
- FX conversion uses the vendor's ~17:00 New York daily fix for London 16:30 closes (documented in
  `quant/data/fx.py`); it is forward-filled, never backfilled.
- Fee and levy rates live in `quant/config/fees.yaml` with `verified` flags; the PTM levy increase and the
  SEC Section 31 rate are seeded from memory and need checking against the primary sources before use.
- Licensed vendors (Bloomberg, Refinitiv, FactSet, Capital IQ) are stubs that document the wiring and raise.

**Models**
- Covariance, VaR and stress numbers describe the sample they were estimated from; parametric and
  simulation VaR assume iid returns and scale with the square root of time.
- Expected returns fed to the optimiser default to shrunk historical means, which is a weak input the
  optimiser amplifies; the result reports binding constraints for that reason.
- Conditioning correlations on a portfolio's own worst days is biased; the breakdown says so in its output.
- The deflated Sharpe ratio counts variants per process, not across your research history.

**Execution**
- SimBroker fills are the engine's idealised fills (next session open, participation cap, half-spread and
  slippage from the cost model); queue position, halts, odd lots and venue outages are not modelled.
- The broker rejects an unaffordable buy at submission, whereas the engine trims it at fill: the two agree
  only while orders are affordable, which the equivalence test guarantees by construction.
- The Alpaca adapter has been exercised only against a fake client; it has not been run against Alpaca's
  paper API from this environment.

**Demo**
- `quant demo` runs on SYNTHETIC data generated from a seeded factor model. Its reports show that the
  pipeline works, not what any market did.
