"""Data-quality validation for daily bars.

For every instrument the report measures: calendar gaps and off-calendar bars, stale (repeated) closes,
return outliers (robust z-score on log returns), OHLC consistency, large jumps classified as a
denomination change (GBX <-> GBP, ~100x), a split explained by a recorded action, or an unexplained jump,
dividend sanity (implied yield), and the share of zero-volume bars for traded instruments. A 0-100 score
and an A/B/C/F grade summarise the result; nothing is corrected silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant.core.calendars import get_calendar
from quant.core.types import AssetClass, Instrument


class ValidationConfig(BaseModel):
    stale_run: int = Field(default=3, ge=2, description="A run of this many equal closes counts as stale.")
    outlier_z: float = Field(default=8.0, gt=0, description="Robust z-score threshold on log returns.")
    jump_ratio: float = Field(default=1.8, gt=1, description="Close/prev-close beyond this is a jump.")
    denomination_band: tuple[float, float] = (90.0, 110.0)
    split_tolerance: float = Field(default=0.25, gt=0, description="Tolerance matching a jump to a split")
    max_dividend_yield: float = Field(default=0.20, gt=0, description="Dividend/price above this is suspect")
    max_listed_issues: int = Field(default=10, ge=1)


@dataclass(frozen=True)
class Issue:
    instrument_id: str
    kind: str
    severity: str  # "error" | "warning"
    date: date | None
    detail: str


@dataclass
class InstrumentQuality:
    instrument_id: str
    first_session: date | None
    last_session: date | None
    bars: int
    expected_sessions: int
    gaps: int = 0
    gap_pct: float = 0.0
    extra_bars: int = 0
    stale_days: int = 0
    max_stale_run: int = 0
    outliers: int = 0
    ohlc_violations: int = 0
    denomination_changes: int = 0
    explained_splits: int = 0
    unexplained_jumps: int = 0
    dividend_warnings: int = 0
    zero_volume_pct: float = 0.0
    score: float = 100.0
    grade: str = "A"
    issues: list[Issue] = field(default_factory=list)

    def as_row(self) -> dict[str, object]:
        row = {k: v for k, v in self.__dict__.items() if k != "issues"}
        row["issues"] = len(self.issues)
        return row


@dataclass
class DataQualityReport:
    results: dict[str, InstrumentQuality]
    config: ValidationConfig

    def to_frame(self) -> pd.DataFrame:
        rows = [q.as_row() for q in self.results.values()]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("score", ascending=True).reset_index(drop=True)

    def issues_frame(self) -> pd.DataFrame:
        rows = [i.__dict__ for q in self.results.values() for i in q.issues]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(Issue.__dataclass_fields__))

    def worst_grade(self) -> str:
        order = {"A": 0, "B": 1, "C": 2, "F": 3}
        return max((q.grade for q in self.results.values()), key=lambda g: order[g], default="A")

    def ok(self, min_grade: str = "C") -> bool:
        order = {"A": 0, "B": 1, "C": 2, "F": 3}
        return order[self.worst_grade()] <= order[min_grade]

    def __getitem__(self, instrument_id: str) -> InstrumentQuality:
        return self.results[instrument_id]


def grade_for(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 75:
        return "B"
    if score >= 50:
        return "C"
    return "F"


def validate_bars(
    bars: pd.DataFrame,
    instruments: Mapping[str, Instrument],
    actions: pd.DataFrame | None = None,
    config: ValidationConfig | None = None,
) -> DataQualityReport:
    """Validate a (multi-instrument) canonical bars frame against each instrument's calendar."""
    cfg = config or ValidationConfig()
    results: dict[str, InstrumentQuality] = {}
    if bars.empty:
        return DataQualityReport(results, cfg)
    for iid, group in bars.groupby("instrument_id", sort=True):
        inst = instruments[iid] if iid in instruments else None
        acts = actions[actions["instrument_id"] == iid] if actions is not None and not actions.empty else None
        results[str(iid)] = _validate_one(str(iid), group.sort_values("session_date"), inst, acts, cfg)
    return DataQualityReport(results, cfg)


def _validate_one(
    iid: str, g: pd.DataFrame, inst: Instrument | None, acts: pd.DataFrame | None, cfg: ValidationConfig
) -> InstrumentQuality:
    issues: list[Issue] = []
    have = pd.DatetimeIndex(g["session_date"]).normalize()
    first, last = have.min().date(), have.max().date()
    q = InstrumentQuality(iid, first, last, bars=len(g), expected_sessions=len(g))

    def add(kind: str, severity: str, d: pd.Timestamp | date | None, detail: str) -> None:
        if sum(1 for i in issues if i.kind == kind) < cfg.max_listed_issues:
            issues.append(Issue(iid, kind, severity, None if d is None else pd.Timestamp(d).date(), detail))

    # --- calendar: gaps and off-calendar bars
    if inst is not None:
        expected = get_calendar(inst.calendar_code).sessions(first, last)
        q.expected_sessions = len(expected)
        gaps = expected.difference(have)
        extra = have.difference(expected)
        q.gaps, q.extra_bars = len(gaps), len(extra)
        q.gap_pct = len(gaps) / max(1, len(expected))
        for d in gaps[: cfg.max_listed_issues]:
            add("gap", "warning", d, "no bar on a trading session")
        for d in extra[: cfg.max_listed_issues]:
            add("extra_bar", "warning", d, "bar on a non-trading day")

    close = g["close"].to_numpy(dtype="float64")
    dates = have.to_numpy()

    # --- OHLC consistency
    o, h, lo = (g[c].to_numpy(dtype="float64") for c in ("open", "high", "low"))
    eps = 1e-9
    bad = (h < lo - eps) | (close > h * (1 + eps)) | (close < lo * (1 - eps))
    bad |= (o > h * (1 + eps)) | (o < lo * (1 - eps))
    bad |= (close <= 0) | (h <= 0) | (lo <= 0) | (o <= 0)
    bad = np.nan_to_num(bad.astype(float), nan=0.0).astype(bool)
    q.ohlc_violations = int(bad.sum())
    for i in np.flatnonzero(bad)[: cfg.max_listed_issues]:
        detail = f"open={o[i]:.4g} high={h[i]:.4g} low={lo[i]:.4g} close={close[i]:.4g}"
        add("ohlc", "error", dates[i], detail)

    # --- stale closes
    if len(close) > 1:
        same = np.isclose(close[1:], close[:-1], rtol=0, atol=1e-12)
        run = 0
        for i, s in enumerate(same):
            if s:
                run += 1
                q.max_stale_run = max(q.max_stale_run, run + 1)
            else:
                if run + 1 >= cfg.stale_run:
                    q.stale_days += run
                    add("stale", "warning", dates[i], f"{run + 1} consecutive identical closes ending here")
                run = 0
        if run + 1 >= cfg.stale_run:
            q.stale_days += run
            add("stale", "warning", dates[-1], f"{run + 1} consecutive identical closes ending the series")

    # --- jumps: denomination change / split / unexplained; then robust outliers on the rest
    if len(close) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = close[1:] / close[:-1]
        lo_b, hi_b = cfg.denomination_band
        denom = ((ratio >= lo_b) & (ratio <= hi_b)) | ((ratio <= 1 / lo_b) & (ratio >= 1 / hi_b))
        jump = ~denom & ((ratio >= cfg.jump_ratio) | (ratio <= 1 / cfg.jump_ratio))
        split_on: dict[pd.Timestamp, float] = {}
        if acts is not None and not acts.empty:
            for ex, kind, val in acts[["ex_date", "action_type", "value"]].itertuples(index=False):
                if kind == "split" and val > 0:
                    split_on[pd.Timestamp(ex).normalize()] = float(val)
        for i in np.flatnonzero(denom):
            q.denomination_changes += 1
            detail = f"close ratio {ratio[i]:.2f}: looks like GBX<->GBP, not a split"
            add("denomination_change", "error", dates[i + 1], detail)
        for i in np.flatnonzero(jump):
            d = pd.Timestamp(dates[i + 1])
            s = split_on.get(d)
            tol = cfg.split_tolerance
            if s is not None and (abs(ratio[i] * s - 1) <= tol or abs(ratio[i] / s - 1) <= tol):
                q.explained_splits += 1
            else:
                q.unexplained_jumps += 1
                add("unexplained_jump", "error", d, f"close ratio {ratio[i]:.3f} with no matching split")
        with np.errstate(divide="ignore", invalid="ignore"):
            logret = np.log(ratio)
        mask = np.isfinite(logret) & ~denom & ~jump
        if mask.sum() >= 20:
            med = np.median(logret[mask])
            mad = np.median(np.abs(logret[mask] - med)) * 1.4826
            if mad > 0:
                z = np.zeros_like(logret)
                z[mask] = (logret[mask] - med) / mad
                out = np.abs(z) > cfg.outlier_z
                q.outliers = int(out.sum())
                for i in np.flatnonzero(out)[: cfg.max_listed_issues]:
                    detail = f"log return {logret[i]:+.3f} is {z[i]:+.1f} robust sigmas"
                    add("outlier", "warning", dates[i + 1], detail)

    # --- dividends
    if acts is not None and not acts.empty:
        divs = acts[acts["action_type"] == "dividend"]
        for ex, val in divs[["ex_date", "value"]].itertuples(index=False):
            ex_ts = pd.Timestamp(ex).normalize()
            prior = close[have < ex_ts]
            ref = prior[-1] if len(prior) else (close[have == ex_ts][0] if (have == ex_ts).any() else np.nan)
            if val <= 0:
                q.dividend_warnings += 1
                add("dividend", "warning", ex_ts, f"non-positive dividend {val}")
            elif np.isfinite(ref) and ref > 0 and val / ref > cfg.max_dividend_yield:
                q.dividend_warnings += 1
                detail = f"dividend {val:.4g} is {val / ref:.0%} of price {ref:.4g}: unit error?"
                add("dividend", "warning", ex_ts, detail)

    # --- volume
    if inst is not None and inst.asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.CRYPTO):
        vol = g["volume"].to_numpy(dtype="float64")
        q.zero_volume_pct = float(np.mean(~(vol > 0))) if len(vol) else 0.0
        if q.zero_volume_pct > 0.05:
            add("volume", "warning", None, f"{q.zero_volume_pct:.0%} of bars have zero volume")

    # --- score
    n = max(1, len(g))
    score = 100.0
    score -= 40 * min(1.0, q.gap_pct / 0.05)
    score -= 15 * min(1.0, q.extra_bars / 5)
    score -= 15 * min(1.0, q.stale_days / max(1.0, 0.05 * n))
    score -= 15 * min(1.0, q.outliers / 5)
    score -= 10 * min(1.0, q.ohlc_violations / 5)
    score -= 20 if (q.denomination_changes or q.unexplained_jumps) else 0
    score -= 10 if q.dividend_warnings else 0
    score -= 10 if q.zero_volume_pct > 0.05 else 0
    q.score = round(max(0.0, score), 1)
    q.grade = grade_for(q.score)
    q.issues = issues
    return q
