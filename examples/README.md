# Examples

Small scripts that use the `quant` Python API directly. They fetch real data through the configured
adapters (network needed on first run; cached afterwards), so they are not part of the test suite — the
offline, fully tested equivalent is `quant demo`, which runs the same pipeline on synthetic data.

| Script | What it does |
|---|---|
| `us_uk_momentum.py` | Cross-sectional momentum across US and UK names, GBP base, with costs, walk-forward and a tearsheet |
| `gilt_curve_regime.py` | UK and US yield-curve slopes, inversion flags and the cross-asset regime board |
| `pairs_ftse.py` | Engle-Granger / Johansen tests on two FTSE names and a cointegration pairs backtest |

Every output ends with a Caveats section. None of these scripts is a recommendation; they show how the
pieces fit together.
