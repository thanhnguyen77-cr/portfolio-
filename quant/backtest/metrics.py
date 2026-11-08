"""Performance metrics from a daily equity curve (base currency), optionally relative to a benchmark.

Annualisation uses 252 periods; CAGR uses the actual calendar span. Sharpe/Sortino subtract a risk-free
rate given either as an annual constant or a daily series aligned to the equity index.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def returns_from_equity(equity: pd.Series) -> pd.Series:
    eq = equity.astype("float64")
    return eq.pct_change().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    eq = equity.astype("float64")
    return eq / eq.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest underwater stretch in sessions)."""
    dd = drawdown_series(equity)
    if dd.empty:
        return 0.0, 0
    underwater = dd < 0
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(dd.min()), int(longest)


def _rf_daily(rf: float | pd.Series | None, index: pd.Index) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=index)
    if isinstance(rf, pd.Series):
        return rf.reindex(index).ffill().fillna(0.0) / PERIODS_PER_YEAR
    return pd.Series(float(rf) / PERIODS_PER_YEAR, index=index)


def compute_metrics(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    rf: float | pd.Series | None = None,
    *,
    turnover: pd.Series | None = None,
    gross_exposure: pd.Series | None = None,
) -> dict[str, Any]:
    eq = equity.astype("float64").dropna()
    out: dict[str, Any] = {"sessions": int(len(eq))}
    if len(eq) < 2:
        out.update({"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": float("nan")})
        return out
    rets = returns_from_equity(eq)
    rf_d = _rf_daily(rf, rets.index)
    excess = rets - rf_d
    years = max((eq.index[-1] - eq.index[0]).days, 1) / 365.25
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    vol = float(rets.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(rets) > 1 else 0.0
    vol = vol if vol > 1e-12 else 0.0  # floating-point noise on a constant path is not volatility
    downside = excess[excess < 0]
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(PERIODS_PER_YEAR)) if len(downside) else 0.0
    downside_dev = downside_dev if downside_dev > 1e-12 else 0.0
    mdd, underwater = max_drawdown(eq)
    cagr = float((1 + total) ** (1 / years) - 1) if total > -1 else -1.0
    mean_excess_ann = float(excess.mean() * PERIODS_PER_YEAR)
    out.update(
        {
            "start": eq.index[0].date(),
            "end": eq.index[-1].date(),
            "years": round(years, 3),
            "total_return": float(total),
            "cagr": cagr,
            "volatility": vol,
            "sharpe": mean_excess_ann / vol if vol > 0 else float("nan"),
            "sortino": mean_excess_ann / downside_dev if downside_dev > 0 else float("nan"),
            "max_drawdown": mdd,
            "longest_underwater_sessions": underwater,
            "calmar": cagr / abs(mdd) if mdd < 0 else float("nan"),
            "hit_rate": float((rets > 0).mean()),
            "best_day": float(rets.max()),
            "worst_day": float(rets.min()),
            "skew": float(rets.skew()) if len(rets) > 2 else float("nan"),
            "kurtosis": float(rets.kurt()) if len(rets) > 3 else float("nan"),
            "final_equity": float(eq.iloc[-1]),
        }
    )
    by_year = (1 + rets).groupby(rets.index.year).prod() - 1
    out["returns_by_year"] = {int(y): float(v) for y, v in by_year.items()}
    if turnover is not None and len(turnover):
        out["avg_daily_turnover"] = float(turnover.mean())
        out["annual_turnover"] = float(turnover.mean() * PERIODS_PER_YEAR)
    if gross_exposure is not None and len(gross_exposure):
        out["avg_gross_exposure"] = float(gross_exposure.mean())
    if benchmark is not None and len(benchmark.dropna()) > 2:
        out.update(benchmark_relative(rets, benchmark, rf_d))
    return out


def benchmark_relative(rets: pd.Series, benchmark: pd.Series, rf_d: pd.Series) -> dict[str, Any]:
    """Beta, annualised alpha, tracking error, information ratio and correlation vs a benchmark."""
    bench = benchmark.astype("float64").dropna()
    bench_rets = bench.pct_change().dropna() if (bench > 0).all() and bench.max() > 1.5 else bench
    joined = pd.concat(
        [rets.rename("p"), bench_rets.rename("b"), rf_d.rename("rf")], axis=1, join="inner"
    ).dropna()
    if len(joined) < 3:
        return {}
    p, b = joined["p"] - joined["rf"], joined["b"] - joined["rf"]
    var_b = float(b.var(ddof=1))
    beta = float(p.cov(b) / var_b) if var_b > 0 else float("nan")
    alpha_daily = float(p.mean() - beta * b.mean()) if np.isfinite(beta) else float("nan")
    active = joined["p"] - joined["b"]
    te = float(active.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    return {
        "beta": beta,
        "alpha_annual": alpha_daily * PERIODS_PER_YEAR if np.isfinite(alpha_daily) else float("nan"),
        "tracking_error": te,
        "information_ratio": float(active.mean() * PERIODS_PER_YEAR / te) if te > 1e-12 else float("nan"),
        "correlation": float(joined["p"].corr(joined["b"])),
        "benchmark_total_return": float((1 + joined["b"]).prod() - 1),
    }
