"""Value at Risk and expected shortfall: historical, parametric (normal / Student-t), Cornish-Fisher,
filtered historical simulation (GARCH) and Monte Carlo, plus Kupiec backtesting of a VaR series.

Conventions: `returns` are decimal period returns; `alpha` is the confidence level (0.99 -> the 1% tail);
VaR and CVaR are reported as POSITIVE loss fractions of the portfolio value over `horizon` periods
(square-root-of-time scaling for the parametric and simulation methods, which assumes iid returns and is
stated in the notes). A VaR number is an estimate from a sample; the Kupiec test says whether a VaR
series has behaved as advertised, not whether it will.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    method: str
    alpha: float
    horizon: int
    var: float
    cvar: float
    n_obs: int
    params: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "horizon": self.horizon,
            "var": self.var,
            "cvar": self.cvar,
            "n_obs": self.n_obs,
            **{f"p_{k}": v for k, v in self.params.items()},
        }


def _clean(returns: pd.Series, min_obs: int = 30) -> np.ndarray:
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} returns, got {len(r)}")
    return r


def _check(alpha: float, horizon: int) -> None:
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1)")
    if horizon < 1:
        raise ValueError("horizon must be >= 1 period")


def _hist_var_cvar(losses: np.ndarray, alpha: float) -> tuple[float, float]:
    """Losses are positive numbers; VaR is the alpha-quantile of losses, CVaR the mean beyond it."""
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, max(cvar, var)


def historical_var(returns: pd.Series, alpha: float = 0.99, horizon: int = 1) -> VaRResult:
    """Empirical quantile of (optionally horizon-aggregated, overlapping) returns."""
    _check(alpha, horizon)
    r = _clean(returns)
    notes: list[str] = []
    if horizon > 1:
        agg = pd.Series(r).rolling(horizon).apply(lambda x: np.prod(1 + x) - 1, raw=True).dropna().to_numpy()
        notes.append(f"{horizon}-period returns built from overlapping windows ({len(agg)} observations)")
        r = agg
    var, cvar = _hist_var_cvar(-r, alpha)
    return VaRResult("historical", alpha, horizon, var, cvar, len(r), {}, notes)


def parametric_var(
    returns: pd.Series, alpha: float = 0.99, horizon: int = 1, dist: str = "normal"
) -> VaRResult:
    """Normal or Student-t VaR/ES from sample moments (t: df fitted by maximum likelihood)."""
    _check(alpha, horizon)
    r = _clean(returns)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    h = math.sqrt(horizon)
    mu_h, sd_h = mu * horizon, sd * h
    notes = ["square-root-of-time scaling assumes iid returns"] if horizon > 1 else []
    if dist == "normal":
        z = stats.norm.ppf(alpha)
        var = -mu_h + z * sd_h
        es = -mu_h + sd_h * stats.norm.pdf(z) / (1 - alpha)
        params = {"mu": mu, "sigma": sd}
        method = "parametric_normal"
    elif dist == "t":
        df, loc, scale = stats.t.fit(r)
        if df <= 2.0:
            notes.append(f"fitted df {df:.2f} <= 2: variance undefined; df floored at 2.05")
            df = 2.05
        tq = stats.t.ppf(alpha, df)
        var = -(loc * horizon) + tq * scale * h
        es_unit = (df + tq**2) / (df - 1) * stats.t.pdf(tq, df) / (1 - alpha)
        es = -(loc * horizon) + es_unit * scale * h
        params = {"mu": float(loc), "scale": float(scale), "df": float(df)}
        method = "parametric_t"
    else:
        raise ValueError("dist must be 'normal' or 't'")
    return VaRResult(method, alpha, horizon, float(var), float(max(es, var)), len(r), params, notes)


def cornish_fisher_z(z: float, skew: float, excess_kurtosis: float) -> float:
    """Cornish-Fisher expansion of a normal quantile for skew and excess kurtosis."""
    return (
        z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * excess_kurtosis / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
    )


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.99, horizon: int = 1) -> VaRResult:
    """Modified (Cornish-Fisher) VaR; ES by numerical integration of the modified quantile function."""
    _check(alpha, horizon)
    r = _clean(returns)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sk = float(stats.skew(r, bias=False))
    ek = float(stats.kurtosis(r, fisher=True, bias=False))
    h = math.sqrt(horizon)
    z_loss = -stats.norm.ppf(1 - alpha)  # positive
    z_cf = -cornish_fisher_z(stats.norm.ppf(1 - alpha), sk, ek)
    var = -mu * horizon + z_cf * sd * h
    ps = np.linspace(1e-6, 1 - alpha, 400)
    q = np.array([cornish_fisher_z(stats.norm.ppf(p), sk, ek) for p in ps])
    es = -mu * horizon - float(np.trapezoid(q, ps) / (1 - alpha)) * sd * h
    notes = [f"normal z {z_loss:.3f} -> modified z {z_cf:.3f} (skew {sk:.2f}, excess kurtosis {ek:.2f})"]
    if abs(sk) > 1.5 or ek > 10:
        notes.append("moments far from normal: the Cornish-Fisher expansion is unreliable here")
    return VaRResult(
        "cornish_fisher",
        alpha,
        horizon,
        float(var),
        float(max(es, var)),
        len(r),
        {"skew": sk, "exkurt": ek},
        notes,
    )


def filtered_historical_var(
    returns: pd.Series, alpha: float = 0.99, horizon: int = 1, **garch_kwargs: object
) -> VaRResult:
    """Filtered historical simulation: GARCH-standardised residuals rescaled by the next-period vol."""
    from quant.stats.garch import fit_garch

    _check(alpha, horizon)
    r = pd.Series(returns, dtype=float).dropna()
    _clean(r, min_obs=100)
    g = fit_garch(r, **garch_kwargs)  # type: ignore[arg-type]
    fc = g._fit.forecast(horizon=1, reindex=False)
    sigma_next = float(np.sqrt(fc.variance.iloc[-1, 0])) / 100.0  # fit_garch scales returns to percent
    mu_next = float(fc.mean.iloc[-1, 0]) / 100.0
    z = g.std_resid.dropna().to_numpy()
    sims = mu_next * horizon + sigma_next * math.sqrt(horizon) * z
    var, cvar = _hist_var_cvar(-sims, alpha)
    notes = [
        f"{g.model}({g.dist}) sigma forecast {sigma_next * math.sqrt(252):.1%} annualised, "
        f"persistence {g.persistence:.3f}, converged={g.converged}"
    ]
    if horizon > 1:
        notes.append("multi-period FHS scales one-step forecasts by sqrt(h); it ignores vol mean reversion")
    return VaRResult(
        "filtered_historical",
        alpha,
        horizon,
        var,
        cvar,
        len(z),
        {"sigma_next_daily": sigma_next, "persistence": g.persistence},
        notes,
    )


def monte_carlo_var(
    weights: pd.Series | Mapping[str, float],
    mu: pd.Series | Mapping[str, float],
    cov: pd.DataFrame,
    alpha: float = 0.99,
    horizon: int = 1,
    n_sims: int = 100_000,
    dist: str = "normal",
    df: float = 5.0,
    seed: int | None = 0,
) -> VaRResult:
    """Portfolio VaR from simulated asset returns (multivariate normal or Student-t with `df`).

    `mu` and `cov` are PER-PERIOD (daily) moments on the same basis as the horizon.
    """
    _check(alpha, horizon)
    w = pd.Series(weights, dtype=float).reindex(cov.index).fillna(0.0).to_numpy()
    m = pd.Series(mu, dtype=float).reindex(cov.index).fillna(0.0).to_numpy()
    sigma = cov.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma + 1e-14 * np.eye(len(w)))
    z = rng.standard_normal((n_sims, len(w)))
    if dist == "t":
        if df <= 2:
            raise ValueError("df must exceed 2")
        g = rng.chisquare(df, n_sims) / df
        z = z / np.sqrt(g)[:, None] * math.sqrt((df - 2) / df)  # unit-variance t
    elif dist != "normal":
        raise ValueError("dist must be 'normal' or 't'")
    asset = m * horizon + (z @ chol.T) * math.sqrt(horizon)
    port = asset @ w
    var, cvar = _hist_var_cvar(-port, alpha)
    notes = [f"{n_sims:,} {dist} simulations, seed {seed}; sqrt-of-time scaling for horizon {horizon}"]
    return VaRResult(f"monte_carlo_{dist}", alpha, horizon, var, cvar, n_sims, {"n_assets": len(w)}, notes)


def var_table(
    returns: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    *,
    include_fhs: bool = True,
) -> pd.DataFrame:
    """One row per method; methods that cannot run on this sample are reported with the reason."""
    rows: list[dict[str, object]] = []
    runners = [
        ("historical", lambda: historical_var(returns, alpha, horizon)),
        ("parametric_normal", lambda: parametric_var(returns, alpha, horizon, "normal")),
        ("parametric_t", lambda: parametric_var(returns, alpha, horizon, "t")),
        ("cornish_fisher", lambda: cornish_fisher_var(returns, alpha, horizon)),
    ]
    if include_fhs:
        runners.append(("filtered_historical", lambda: filtered_historical_var(returns, alpha, horizon)))
    for name, fn in runners:
        try:
            res = fn()
            rows.append({**res.as_row(), "notes": "; ".join(res.notes)})
        except Exception as exc:  # noqa: BLE001 - a failing estimator is reported, not fatal
            rows.append(
                {"method": name, "alpha": alpha, "horizon": horizon, "var": np.nan, "cvar": np.nan,
                 "n_obs": 0, "notes": f"failed: {type(exc).__name__}: {str(exc)[:120]}"}
            )  # fmt: skip
    cols = ["method", "alpha", "horizon", "var", "cvar", "n_obs", "notes"]
    df = pd.DataFrame(rows)
    return df[cols + [c for c in df.columns if c not in cols]]


def rolling_historical_var(returns: pd.Series, window: int = 250, alpha: float = 0.99) -> pd.Series:
    """Out-of-sample VaR for each period from the previous `window` returns (shifted: no look-ahead)."""
    r = pd.Series(returns, dtype=float)
    var = r.rolling(window).quantile(1 - alpha).shift(1)
    return (-var).rename("var")


def kupiec_test(returns: pd.Series, var: pd.Series, alpha: float = 0.99) -> dict[str, float | int | bool]:
    """Kupiec proportion-of-failures test: are exceptions (loss > VaR) as frequent as 1 - alpha implies?"""
    data = pd.concat(
        [pd.Series(returns, dtype=float).rename("r"), pd.Series(var, dtype=float).rename("v")], axis=1
    )
    data = data.dropna()
    n = len(data)
    if n == 0:
        raise ValueError("no overlapping observations between returns and VaR")
    x = int((-data["r"] > data["v"]).sum())
    p = 1 - alpha
    p_hat = x / n
    if x == 0:
        lr = -2 * n * math.log(1 - p)
    elif x == n:
        lr = -2 * n * math.log(p)
    else:
        lr = -2 * (
            x * math.log(p) + (n - x) * math.log(1 - p) - x * math.log(p_hat) - (n - x) * math.log(1 - p_hat)
        )
    p_value = float(1 - stats.chi2.cdf(lr, df=1))
    return {
        "n": n,
        "exceptions": x,
        "expected": n * p,
        "exception_rate": p_hat,
        "lr_pof": float(lr),
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
    }
