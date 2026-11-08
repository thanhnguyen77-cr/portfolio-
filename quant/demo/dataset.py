"""A SYNTHETIC mini-dataset for the offline end-to-end demo.

Twenty made-up equities (ten on Nasdaq in USD, ten on the LSE quoted in pence), index and volatility proxies
under the catalogue's index ids, a GBPUSD path, a planted 4:1 split, quarterly dividends, and a handful of
macro series. Prices are a seeded factor model (market + sector + idiosyncratic noise) on the real exchange
calendars. The data exercises every stage of the pipeline; it says nothing about any real market, and the
vendor label `demo` keeps it from ever being mistaken for vendor data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant.core.calendars import get_calendar
from quant.core.time import now
from quant.core.types import (
    BAR_COLUMNS,
    AdjustmentState,
    AssetClass,
    Currency,
    Instrument,
    Market,
    StampDutyClass,
)
from quant.data.adapters.base import (
    CORPORATE_ACTION_COLUMNS,
    AdapterResult,
    BaseAdapter,
    Capability,
    empty_bars,
)
from quant.data.service import DataService

DEMO_VENDOR = "demo"
DEMO_SEED = 42
DEMO_START = date(2020, 1, 2)
DEMO_END = date(2024, 6, 28)
SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA: every price, action and macro observation in this dataset was generated from a seeded "
    "model. Results describe the generator, not any market."
)

US_NAMES = [
    ("US01", "Technology"),
    ("US02", "Technology"),
    ("US03", "Financials"),
    ("US04", "Financials"),
    ("US05", "Energy"),
    ("US06", "Health Care"),
    ("US07", "Consumer"),
    ("US08", "Industrials"),
    ("US09", "Utilities"),
    ("US10", "Materials"),
]
UK_NAMES = [
    ("UK01", "Financials"),
    ("UK02", "Energy"),
    ("UK03", "Consumer"),
    ("UK04", "Health Care"),
    ("UK05", "Materials"),
    ("UK06", "Industrials"),
    ("UK07", "Telecommunications"),
    ("UK08", "Utilities"),
    ("UK09", "Financials"),
    ("UK10", "Consumer"),
]
INDEX_IDS = {"us_index": "XNYS_SPX", "us_vol": "XNYS_VIX", "uk_index": "XLON_UKX", "fx": "FX_GBPUSD"}
SPLIT = ("XNAS_US01", date(2021, 6, 1), 4.0)
DIVIDEND_PAYERS = {"XNAS_US03": 0.50, "XLON_UK01": 0.05}


class DemoAdapter(BaseAdapter):
    """Serves nothing: the dataset is static. Exists so `--vendor demo` resolves and lineage stays honest."""

    name = DEMO_VENDOR
    identifier_key = "demo"

    def capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.BARS_DAILY})

    def get_bars(
        self, instruments: Sequence[Instrument], start: date, end: date, freq: str = "1d"
    ) -> AdapterResult:
        ids = [i.instrument_id for i in instruments]
        return AdapterResult(
            vendor=self.name,
            frame=empty_bars(),
            errors={i: "synthetic demo dataset is static; run `quant demo` to regenerate it" for i in ids},
            requested=ids,
            params={"start": str(start), "end": str(end)},
        )


def demo_instruments() -> list[Instrument]:
    out = []
    for sym, sector in US_NAMES:
        out.append(
            Instrument(
                symbol=sym,
                name=f"Demo {sym} ({sector})",
                asset_class=AssetClass.EQUITY,
                market=Market.XNAS,
                currency=Currency.USD,
                country="US",
                sector=sector,
                stamp_duty_class=StampDutyClass.EXEMPT,
                identifiers={"demo": sym},
                notes="synthetic demo instrument",
            )
        )
    for sym, sector in UK_NAMES:
        out.append(
            Instrument(
                symbol=sym,
                name=f"Demo {sym} ({sector})",
                asset_class=AssetClass.EQUITY,
                market=Market.XLON,
                currency=Currency.GBP,
                price_scale=0.01,
                country="GB",
                sector=sector,
                stamp_duty_class=StampDutyClass.UK_ORDINARY,
                identifiers={"demo": sym},
                notes="synthetic demo instrument (quoted in pence)",
            )
        )
    return out


@dataclass
class DemoManifest:
    seed: int
    start: date
    end: date
    vendor: str = DEMO_VENDOR
    instruments: list[str] = field(default_factory=list)
    universes: dict[str, list[str]] = field(default_factory=dict)
    actions: int = 0
    macro_series: list[str] = field(default_factory=list)
    notice: str = SYNTHETIC_NOTICE
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "vendor": self.vendor,
            "instruments": self.instruments,
            "universes": self.universes,
            "actions": self.actions,
            "macro_series": self.macro_series,
            "notice": self.notice,
            "generated_at": self.generated_at,
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def _ohlcv(
    inst: Instrument,
    sessions: pd.DatetimeIndex,
    close: np.ndarray,
    rng: np.random.Generator,
    volume_base: float,
) -> pd.DataFrame:
    n = len(sessions)
    noise = rng.normal(0, 0.004, n)
    open_ = np.r_[close[0], close[:-1]] * (1 + noise)
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    volume = np.exp(rng.normal(np.log(volume_base), 0.35, n)).round()
    cal = get_calendar(inst.calendar_code)
    closes_utc = cal.session_closes(sessions[0].date(), sessions[-1].date())
    df = pd.DataFrame(
        {
            "instrument_id": inst.instrument_id,
            "session_date": sessions,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "currency": inst.currency.value,
            "vendor": DEMO_VENDOR,
            "adjustment_state": AdjustmentState.RAW.value,
            "ts_close_utc": pd.Series(sessions).map(closes_utc).to_numpy(),
            "ingested_at": pd.Timestamp(now()),
        }
    )
    return df.loc[:, list(BAR_COLUMNS)]


def generate_bars(
    instruments: Sequence[Instrument], start: date, end: date, seed: int = DEMO_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """(bars for equities, indices, vol index and FX; corporate actions; the factor paths used)."""
    rng = np.random.default_rng(seed)
    cals = {"XNAS": get_calendar("XNAS"), "XLON": get_calendar("XLON"), "FX": get_calendar("FX")}
    sess = {code: cal.sessions(start, end) for code, cal in cals.items()}
    us_mkt = rng.normal(0.00035, 0.011, len(sess["XNAS"]))
    uk_mkt = 0.6 * np.interp(
        np.arange(len(sess["XLON"])), np.linspace(0, len(sess["XLON"]) - 1, len(us_mkt)), us_mkt
    )
    uk_mkt = uk_mkt + rng.normal(0.0001, 0.008, len(sess["XLON"]))
    sectors = sorted({s for _, s in US_NAMES + UK_NAMES})
    sector_paths = {s: rng.normal(0, 0.006, max(len(sess["XNAS"]), len(sess["XLON"]))) for s in sectors}
    frames: list[pd.DataFrame] = []
    actions: list[tuple[str, date, str, float, str | None]] = []
    by_id = {i.instrument_id: i for i in instruments}
    for inst in instruments:
        code = inst.calendar_code
        s = sess[code]
        mkt = us_mkt if code == "XNAS" else uk_mkt
        beta = rng.uniform(0.6, 1.4)
        sec = sector_paths[inst.sector or "unknown"][: len(s)]
        idio_vol = rng.uniform(0.008, 0.02)
        drift = rng.normal(0.0001, 0.00025)
        rets = drift + beta * mkt + 0.7 * sec + rng.normal(0, idio_vol, len(s))
        start_price = float(rng.uniform(20, 300)) if code == "XNAS" else float(rng.uniform(0.8, 30.0))
        close = start_price * np.exp(np.cumsum(rets))
        if inst.instrument_id == SPLIT[0]:  # planted 4:1 split: raw prices quarter from the ex-date on
            ex = pd.Timestamp(SPLIT[1])
            close = np.where(s >= ex, close / SPLIT[2], close)
            actions.append((inst.instrument_id, SPLIT[1], "split", SPLIT[2], None))
        if inst.instrument_id in DIVIDEND_PAYERS:
            for q in pd.date_range(start, end, freq="QS-MAR"):
                ex_day = s[s >= q]
                if len(ex_day):
                    actions.append(
                        (
                            inst.instrument_id,
                            ex_day[0].date(),
                            "dividend",
                            DIVIDEND_PAYERS[inst.instrument_id],
                            inst.currency.value,
                        )
                    )
        frames.append(_ohlcv(inst, s, close, rng, volume_base=2e6 if code == "XNAS" else 8e6))
    # indices, vol proxy and FX under the catalogue's ids (tradable=False in the catalogue)
    from quant.core.catalog import load_default_catalog

    cat = load_default_catalog()
    spx = 3200 * np.exp(np.cumsum(us_mkt))
    frames.append(_ohlcv(cat[INDEX_IDS["us_index"]], sess["XNAS"], spx, rng, 1.0))
    ukx = 7400 * np.exp(np.cumsum(uk_mkt))
    frames.append(_ohlcv(cat[INDEX_IDS["uk_index"]], sess["XLON"], ukx, rng, 1.0))
    rv = pd.Series(us_mkt).rolling(21, min_periods=5).std().bfill().to_numpy() * np.sqrt(252)
    vix = np.clip(100 * rv * 1.1 + rng.normal(0, 1.0, len(rv)) + 4, 9, 90)
    frames.append(_ohlcv(cat[INDEX_IDS["us_vol"]], sess["XNAS"], vix, rng, 1.0))
    fx = 1.27 * np.exp(np.cumsum(rng.normal(0, 0.0055, len(sess["FX"]))))
    frames.append(_ohlcv(cat[INDEX_IDS["fx"]], sess["FX"], fx, rng, 1.0))
    bars = pd.concat(frames, ignore_index=True)
    acts = pd.DataFrame(actions, columns=["instrument_id", "ex_date", "action_type", "value", "currency"])
    acts["ex_date"] = pd.to_datetime(acts["ex_date"])
    acts["vendor"] = DEMO_VENDOR
    acts["recorded_at"] = pd.Timestamp(now())
    acts = acts.loc[:, list(CORPORATE_ACTION_COLUMNS)]
    del by_id
    return (
        bars,
        acts,
        {
            "us_market": pd.Series(us_mkt, index=sess["XNAS"]),
            "uk_market": pd.Series(uk_mkt, index=sess["XLON"]),
        },
    )


def generate_macro(start: date, end: date, seed: int = DEMO_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    days = pd.bdate_range(start, end)
    months = pd.date_range(start, end, freq="MS")
    rows = []

    def add(series_id: str, idx: pd.DatetimeIndex, values: np.ndarray, vendor: str = "fred") -> None:
        rows.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "date": idx,
                    "value": values,
                    "vendor": vendor,
                    "ingested_at": pd.Timestamp(now()),
                }
            )
        )

    policy = np.clip(0.25 + np.cumsum(rng.normal(0.004, 0.02, len(days))), 0.05, 6.0)
    add("DFF", days, policy)
    add("DGS2", days, policy + 0.3 + np.cumsum(rng.normal(0, 0.01, len(days))))
    add("DGS10", days, policy + 0.6 + np.cumsum(rng.normal(0, 0.012, len(days))))
    add("DFII10", days, 0.2 + np.cumsum(rng.normal(0, 0.01, len(days))))
    add("BAMLH0A0HYM2", days, np.clip(3.5 + np.cumsum(rng.normal(0, 0.03, len(days))), 2.0, 12.0))
    add("DTWEXBGS", days, 115 * np.exp(np.cumsum(rng.normal(0, 0.002, len(days)))))
    add("CPIAUCSL", months, 258 * np.exp(np.cumsum(rng.normal(0.003, 0.0015, len(months)))))
    add("UNRATE", months, np.clip(3.6 + np.cumsum(rng.normal(0, 0.1, len(months))), 3.0, 12.0))
    add("INDPRO", months, 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.004, len(months)))))
    add("PAYEMS", months, 150_000 + np.cumsum(rng.normal(150, 120, len(months))))
    add(
        "IUDBEDR", days, np.clip(0.1 + np.cumsum(rng.normal(0.004, 0.015, len(days))), 0.1, 6.0), vendor="boe"
    )
    return pd.concat(rows, ignore_index=True)


def build_demo_dataset(
    service: DataService, *, seed: int = DEMO_SEED, start: date = DEMO_START, end: date = DEMO_END
) -> DemoManifest:
    """Register the synthetic instruments, write bars, actions and macro series, record the universes."""
    insts = demo_instruments()
    service.register(insts)
    if DEMO_VENDOR not in service.adapters:
        service.adapters[DEMO_VENDOR] = DemoAdapter()
    bars, actions, _ = generate_bars(insts, start, end, seed)
    service.parquet.write_bars(bars, service.freq)
    n_actions = service.db.upsert_corporate_actions(actions)
    for iid in sorted(bars["instrument_id"].unique()):
        series = service.parquet.read_series(iid, DEMO_VENDOR, service.freq)
        service.db.upsert_series_meta(
            iid,
            DEMO_VENDOR,
            service.freq,
            first_session=series["session_date"].min().date(),
            last_session=series["session_date"].max().date(),
            rows=len(series),
            written=True,
        )
    macro = generate_macro(start, end, seed)
    service.db.upsert_macro(macro)
    us = [i.instrument_id for i in insts if i.market is Market.XNAS]
    uk = [i.instrument_id for i in insts if i.market is Market.XLON]
    universes = {"demo_us": us, "demo_uk": uk, "demo_all": us + uk}
    for name, ids in universes.items():
        uni = service.universe(name)
        if uni.snapshots().empty:
            uni.add_snapshot(end, ids, source="synthetic demo")
        uni.set_quality(
            "current_only", "synthetic demo: membership fixed by construction", history_start=start
        )
    manifest = DemoManifest(
        seed=seed,
        start=start,
        end=end,
        instruments=sorted(bars["instrument_id"].unique()),
        universes=universes,
        actions=int(n_actions),
        macro_series=sorted(macro["series_id"].unique()),
        generated_at=now().isoformat(),
    )
    manifest.write(service.settings.paths.data_dir / "demo_manifest.json")
    return manifest
