"""Evidence-based universe screening.

The screen selects instruments that are *tradable and well measured*: liquid enough for the intended
size, priced above penny territory, with realised volatility inside a workable band and data that passed
validation. It ranks candidates by a transparent 0-100 score whose components are reported alongside the
raw evidence. It does not predict returns and is not a signal.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core.types import Instrument
from quant.data.fx import FXRates
from quant.data.validation import DataQualityReport

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "F": 3}


class ScreenConfig(BaseModel):
    adv_window: int = Field(default=63, ge=5, description="Sessions for median daily traded value.")
    vol_window: int = Field(default=63, ge=5)
    min_adv_base: float = Field(
        default=5e6, gt=0, description="Minimum median daily traded value in base ccy."
    )
    min_price_base: float = Field(default=1.0, gt=0)
    vol_min: float = Field(default=0.08, gt=0, description="Annualised realised vol lower bound.")
    vol_max: float = Field(default=0.80, gt=0)
    max_range_pct: float = Field(
        default=0.12, gt=0, description="Mean (high-low)/close cap; a spread/noise proxy."
    )
    min_sessions: int = Field(default=200, ge=20)
    min_dq_grade: str = Field(default="C")
    weights: dict[str, float] = Field(
        default_factory=lambda: {"liquidity": 40.0, "quality": 30.0, "volatility": 20.0, "price": 10.0}
    )


def screen_universe(
    bars: pd.DataFrame,
    instruments: Mapping[str, Instrument],
    *,
    base_currency: str = "USD",
    fx: FXRates | None = None,
    data_quality: DataQualityReport | None = None,
    config: ScreenConfig | None = None,
) -> pd.DataFrame:
    """Score and rank instruments present in `bars`. Returns one row per instrument, best first."""
    cfg = config or ScreenConfig()
    rows: list[dict[str, object]] = []
    for iid, g in bars.groupby("instrument_id", sort=True):
        inst = instruments[iid] if iid in instruments else None
        if inst is None:
            continue
        g = g.sort_values("session_date")
        ccy = inst.currency.value
        dates = pd.DatetimeIndex(g["session_date"])
        if ccy != base_currency:
            if fx is None:
                raise ValueError(f"FX rates needed to express {iid} ({ccy}) in {base_currency}")
            rate = fx.rate(ccy, base_currency, dates).to_numpy()
        else:
            rate = np.ones(len(g))
        close = g["close"].to_numpy(dtype="float64")
        volume = np.nan_to_num(g["volume"].to_numpy(dtype="float64"))
        traded_base = close * volume * rate
        adv = float(np.nanmedian(traded_base[-cfg.adv_window :])) if len(g) else 0.0
        price_local = float(close[-1])
        price_base = float(price_local * (rate[-1] if np.isfinite(rate[-1]) else np.nan))
        logret = np.diff(np.log(close[-(cfg.vol_window + 1) :]))
        vol_ann = float(np.std(logret, ddof=1) * np.sqrt(252)) if len(logret) > 2 else float("nan")
        rng = (g["high"].to_numpy(dtype="float64") - g["low"].to_numpy(dtype="float64")) / close
        range_pct = float(np.nanmean(rng[-cfg.adv_window :]))
        sessions = int(len(g))
        dq_grade, dq_score = None, None
        if data_quality is not None and iid in data_quality.results:
            dq_grade, dq_score = data_quality[iid].grade, data_quality[iid].score

        reasons: list[str] = []
        if not adv >= cfg.min_adv_base:
            reasons.append(f"adv {adv:,.0f} < {cfg.min_adv_base:,.0f}")
        if not price_base >= cfg.min_price_base:
            reasons.append(f"price {price_base:.2f} < {cfg.min_price_base}")
        if not (cfg.vol_min <= vol_ann <= cfg.vol_max):
            reasons.append(f"vol {vol_ann:.0%} outside [{cfg.vol_min:.0%}, {cfg.vol_max:.0%}]")
        if not range_pct <= cfg.max_range_pct:
            reasons.append(f"range {range_pct:.1%} > {cfg.max_range_pct:.0%}")
        if sessions < cfg.min_sessions:
            reasons.append(f"history {sessions} < {cfg.min_sessions} sessions")
        if dq_grade is not None and GRADE_ORDER[dq_grade] > GRADE_ORDER[cfg.min_dq_grade]:
            reasons.append(f"data quality {dq_grade} worse than {cfg.min_dq_grade}")

        w = cfg.weights
        liquidity_pts = w["liquidity"] * float(
            np.clip(np.log10(max(adv, 1e-9) / cfg.min_adv_base) / 2.0, 0.0, 1.0)
        )
        if dq_score is not None:
            quality_pts = w["quality"] * dq_score / 100.0
        else:
            quality_pts = (
                w["quality"] * min(1.0, sessions / max(cfg.min_sessions, 1)) * 0.5
            )  # unvalidated: half credit
        centre = np.sqrt(cfg.vol_min * cfg.vol_max)
        half = (np.log(cfg.vol_max) - np.log(cfg.vol_min)) / 2.0
        if np.isfinite(vol_ann) and vol_ann > 0:
            volatility_pts = w["volatility"] * float(
                np.clip(1.0 - abs(np.log(vol_ann) - np.log(centre)) / half, 0.0, 1.0)
            )
        else:
            volatility_pts = 0.0
        price_pts = w["price"] if price_base >= cfg.min_price_base else 0.0
        score = round(liquidity_pts + quality_pts + volatility_pts + price_pts, 1)
        rows.append(
            {
                "instrument_id": iid,
                "symbol": inst.symbol,
                "name": inst.name,
                "currency": ccy,
                "price_local": price_local,
                "price_base": price_base,
                "adv_base": adv,
                "vol_ann": vol_ann,
                "range_pct": range_pct,
                "sessions": sessions,
                "dq_grade": dq_grade,
                "dq_score": dq_score,
                "liquidity_pts": round(liquidity_pts, 1),
                "quality_pts": round(quality_pts, 1),
                "volatility_pts": round(volatility_pts, 1),
                "price_pts": round(price_pts, 1),
                "score": score,
                "passes": not reasons,
                "reasons": "; ".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["passes", "score"], ascending=[False, False]).reset_index(drop=True)
