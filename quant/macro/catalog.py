"""Named macro series. `verified=False` marks ids that have not yet been checked against the provider;
the dashboard reports which series it could not load rather than guessing. Every FRED, BoE and ONS id
below was fetched live on 2026-11-08 (`quant macro verify`); re-run that command after provider changes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroSeries:
    name: str
    vendor: str  # fred | boe | ons
    series_id: str
    frequency: str  # daily | monthly | quarterly
    unit: str  # percent | index | level | bps
    region: str  # us | uk
    description: str
    verified: bool = True


_S = MacroSeries
MACRO_SERIES: dict[str, MacroSeries] = {
    # United States (FRED)
    "us_cpi": _S("us_cpi", "fred", "CPIAUCSL", "monthly", "index", "us", "CPI all items, SA"),
    "us_core_cpi": _S(
        "us_core_cpi", "fred", "CPILFESL", "monthly", "index", "us", "CPI ex food and energy, SA"
    ),
    "us_pce": _S("us_pce", "fred", "PCEPI", "monthly", "index", "us", "PCE price index"),
    "us_unemployment": _S(
        "us_unemployment", "fred", "UNRATE", "monthly", "percent", "us", "Unemployment rate"
    ),
    "us_payrolls": _S(
        "us_payrolls", "fred", "PAYEMS", "monthly", "level", "us", "Nonfarm payrolls (thousands)"
    ),
    "us_industrial_production": _S(
        "us_industrial_production", "fred", "INDPRO", "monthly", "index", "us", "Industrial production"
    ),
    "us_real_gdp": _S("us_real_gdp", "fred", "GDPC1", "quarterly", "level", "us", "Real GDP"),
    "us_fed_funds": _S("us_fed_funds", "fred", "DFF", "daily", "percent", "us", "Effective fed funds rate"),
    "us_3m": _S("us_3m", "fred", "DGS3MO", "daily", "percent", "us", "3-month Treasury"),
    "us_2y": _S("us_2y", "fred", "DGS2", "daily", "percent", "us", "2-year Treasury"),
    "us_5y": _S("us_5y", "fred", "DGS5", "daily", "percent", "us", "5-year Treasury"),
    "us_10y": _S("us_10y", "fred", "DGS10", "daily", "percent", "us", "10-year Treasury"),
    "us_30y": _S("us_30y", "fred", "DGS30", "daily", "percent", "us", "30-year Treasury"),
    "us_real_10y": _S("us_real_10y", "fred", "DFII10", "daily", "percent", "us", "10-year TIPS real yield"),
    "us_breakeven_10y": _S(
        "us_breakeven_10y", "fred", "T10YIE", "daily", "percent", "us", "10-year breakeven inflation"
    ),
    "us_hy_oas": _S(
        "us_hy_oas", "fred", "BAMLH0A0HYM2", "daily", "percent", "us", "ICE BofA US High Yield OAS"
    ),
    "us_ig_oas": _S("us_ig_oas", "fred", "BAMLC0A0CM", "daily", "percent", "us", "ICE BofA US Corporate OAS"),
    "us_dollar_index": _S(
        "us_dollar_index", "fred", "DTWEXBGS", "daily", "index", "us", "Trade-weighted dollar (broad)"
    ),
    "us_sahm": _S(
        "us_sahm", "fred", "SAHMREALTIME", "monthly", "percent", "us", "Sahm rule recession indicator"
    ),
    # United Kingdom (Bank of England / ONS)
    "uk_bank_rate": _S("uk_bank_rate", "boe", "IUDBEDR", "daily", "percent", "uk", "Bank Rate"),
    "uk_sonia": _S(
        "uk_sonia", "boe", "IUDSOIA", "daily", "percent", "uk", "SONIA overnight rate", verified=True
    ),
    "uk_10y": _S(
        "uk_10y", "boe", "IUDMNPY", "daily", "percent", "uk", "10-year gilt nominal par yield", verified=True
    ),
    "uk_5y": _S(
        "uk_5y", "boe", "IUDSNPY", "daily", "percent", "uk", "5-year gilt nominal par yield", verified=True
    ),
    "uk_20y": _S(
        "uk_20y", "boe", "IUDLNPY", "daily", "percent", "uk", "20-year gilt nominal par yield", verified=True
    ),
    "uk_cpih": _S("uk_cpih", "ons", "MM23/L55O", "monthly", "percent", "uk", "CPIH annual rate"),
    "uk_cpi": _S("uk_cpi", "ons", "MM23/D7G7", "monthly", "percent", "uk", "CPI annual rate"),
    "uk_cpi_index": _S(
        "uk_cpi_index", "ons", "MM23/D7BT", "monthly", "index", "uk", "CPI index (2015=100)", verified=True
    ),
    "uk_unemployment": _S(
        "uk_unemployment",
        "ons",
        "/employmentandlabourmarket/peoplenotinwork/unemployment/timeseries/mgsx/lms",
        "monthly",
        "percent",
        "uk",
        "Unemployment rate (16+)",
    ),
    "uk_gdp_monthly": _S(
        "uk_gdp_monthly", "ons", "MGDP/ECY2", "monthly", "index", "uk", "Monthly GDP index", verified=True
    ),
    "uk_retail_sales": _S(
        "uk_retail_sales",
        "ons",
        "DRSI/J5EK",
        "monthly",
        "index",
        "uk",
        "Retail sales volume index",
        verified=True,
    ),
    "uk_employment_rate": _S(
        "uk_employment_rate",
        "ons",
        "/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/timeseries/lf24/lms",
        "monthly",
        "percent",
        "uk",
        "Employment rate, 16-64, SA",
    ),
}


def macro_series_spec(name: str) -> MacroSeries:
    if name not in MACRO_SERIES:
        raise KeyError(f"unknown macro series {name!r}; known: {sorted(MACRO_SERIES)}")
    return MACRO_SERIES[name]


def series_for_region(region: str) -> list[MacroSeries]:
    return [s for s in MACRO_SERIES.values() if region == "both" or s.region == region]
