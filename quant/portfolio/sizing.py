"""Position sizing: fractional Kelly with Wilson shrinkage and ordered caps (ported from the trader plugin's
kelly.py; semantics preserved, re-implemented for base-currency portfolios).

Kelly for a trade with a hard stop: lose 1R (the stop) or win `payoff` R, with win probability p, so
f* = p - (1 - p) / payoff. Full Kelly is far too aggressive for an *estimated* edge, so `size_position`
  1. shrinks p to the 1-sigma Wilson lower bound for the sample size,
  2. multiplies by `kelly_multiplier` (default quarter Kelly),
  3. scales by the regime multiplier (risk_on 1.0, caution 0.5, risk_off 0),
  4. caps the amount at risk at `max_risk_per_trade_pct` of capital,
  5. caps the position value at `max_position_value`,
  6. caps the cost at the cash above `cash_reserve`,
  7. caps total exposure at `max_total_exposure_pct`,
  8. rounds down to whole lots.
The smallest cap wins and is reported as `binding_constraint`; a non-positive Kelly means NO_TRADE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from quant.backtest.fills import round_quantity

REGIME_MULTIPLIER = {"risk_on": 1.0, "caution": 0.5, "risk_off": 0.0}


def kelly_fraction(p: float, b: float) -> float:
    """Fraction of capital to risk for win probability p and payoff ratio b (negative = no edge)."""
    if b <= 0:
        return -1.0
    return p - (1.0 - p) / b


def continuous_kelly(mu_excess: float, variance: float) -> float:
    """Kelly leverage for a Gaussian return stream: mu / sigma^2 (annual inputs)."""
    return mu_excess / variance if variance > 0 else float("nan")


def wilson_lower_bound(p_hat: float, n: int, z: float = 1.0) -> float:
    """Lower bound of the Wilson score interval for a proportion."""
    if n <= 0:
        return 0.0
    denom = 1.0 + z * z / n
    centre = p_hat + z * z / (2 * n)
    margin = z * math.sqrt(max(p_hat * (1 - p_hat) / n + z * z / (4 * n * n), 0.0))
    return max(0.0, (centre - margin) / denom)


@dataclass
class SizingResult:
    decision: str  # SIZE | NO_TRADE
    qty: float
    binding_constraint: str
    caps_qty: dict[str, float]
    kelly_full_raw: float
    kelly_full_conservative: float
    kelly_used: float
    risk_per_share: float
    position_value: float
    amount_at_risk: float
    risk_pct_of_capital: float | None
    cash_after: float | None
    notes: list[str]
    inputs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def size_position(
    *,
    win_rate: float,
    payoff: float,
    entry: float,
    stop: float,
    capital: float,
    n_trades: int = 0,
    cash: float | None = None,
    invested: float = 0.0,
    kelly_multiplier: float = 0.25,
    max_risk_per_trade_pct: float = 1.0,
    max_position_value: float | None = None,
    cash_reserve: float = 0.0,
    max_total_exposure_pct: float = 100.0,
    regime: str = "risk_on",
    conservative: bool = True,
    round_trip_cost_bps: float = 0.0,
    target: float | None = None,
    lot_size: int = 1,
    fractional: bool = False,
) -> SizingResult:
    """Long trade sizing in one currency (`entry`, `stop`, `capital`, `cash` all in that currency)."""
    notes: list[str] = []
    if entry <= 0 or stop <= 0:
        raise ValueError("entry and stop must be positive")
    if stop >= entry:
        raise ValueError("long trades only: stop must be below entry")
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0, 1]")
    if capital <= 0:
        raise ValueError("capital must be positive")

    p_used = wilson_lower_bound(win_rate, n_trades) if (conservative and n_trades > 0) else win_rate
    if conservative and n_trades > 0:
        notes.append(
            f"win rate shrunk from {win_rate:.3f} to {p_used:.3f} (1-sigma lower bound, n={n_trades})"
        )
    f_cons = kelly_fraction(p_used, payoff)
    f_raw = kelly_fraction(win_rate, payoff)
    mult = REGIME_MULTIPLIER.get(regime, 0.5)
    f_used = max(0.0, f_cons) * float(kelly_multiplier) * mult
    risk_per_share = (entry - stop) + entry * round_trip_cost_bps / 1e4
    available_cash = cash if cash is not None else capital - invested
    inputs = {
        "win_rate": win_rate,
        "win_rate_used": p_used,
        "payoff": payoff,
        "n_trades": n_trades,
        "entry": entry,
        "stop": stop,
        "target": target,
        "capital": capital,
        "cash": available_cash,
        "invested": invested,
        "regime": regime,
        "regime_multiplier": mult,
        "kelly_multiplier": kelly_multiplier,
    }

    def no_trade(reason: str, binding: str) -> SizingResult:
        return SizingResult(
            "NO_TRADE", 0.0, binding, {}, f_raw, f_cons, f_used, risk_per_share, 0.0, 0.0, 0.0,
            available_cash, notes + [reason], inputs
        )  # fmt: skip

    if f_cons <= 0:
        return no_trade(f"Kelly is {f_cons:.3f} <= 0: the measured edge does not justify any risk", "kelly")
    if mult == 0:
        return no_trade("market regime is risk_off: no new longs", "regime")

    def lots(x: float) -> float:
        return round_quantity(max(x, 0.0), lot_size, fractional)

    caps: dict[str, float] = {
        "kelly": lots(capital * f_used / risk_per_share),
        "max_risk_per_trade": lots(capital * max_risk_per_trade_pct / 100.0 / risk_per_share),
        "cash_reserve": lots(max(available_cash - cash_reserve, 0.0) / entry),
    }
    if max_position_value is not None:
        caps["max_position_value"] = lots(max_position_value / entry)
    exposure_room = capital * max_total_exposure_pct / 100.0 - invested
    caps["max_total_exposure"] = lots(max(exposure_room, 0.0) / entry)
    binding = min(caps, key=lambda k: (caps[k], k))
    qty = caps[binding]
    position_value = qty * entry
    at_risk = qty * risk_per_share
    if qty <= 0:
        notes.append(
            f"position rounds to zero (binding: {binding}); the stop is too wide or the limits too tight"
        )
        decision = "NO_TRADE"
    else:
        decision = "SIZE"
        if binding != "kelly":
            notes.append(f"Kelly would allow {caps['kelly']:g}; capped at {qty:g} by {binding}")
    if target is not None and target > entry:
        inputs["reward_risk"] = (target - entry) / (entry - stop)
        inputs["potential_profit"] = qty * (target - entry)
    return SizingResult(
        decision=decision,
        qty=qty,
        binding_constraint=binding,
        caps_qty=caps,
        kelly_full_raw=f_raw,
        kelly_full_conservative=f_cons,
        kelly_used=f_used,
        risk_per_share=risk_per_share,
        position_value=position_value,
        amount_at_risk=at_risk,
        risk_pct_of_capital=at_risk / capital * 100.0,
        cash_after=available_cash - position_value,
        notes=notes,
        inputs=inputs,
    )
