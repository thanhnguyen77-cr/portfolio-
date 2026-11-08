"""BacktestResult: everything a run produced, plus the caveats that must travel with it."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BacktestResult:
    strategy: dict[str, Any]
    config: dict[str, Any]
    equity: (
        pd.DataFrame
    )  # session_date, equity, cash_base, positions_value, gross_exposure, net_exposure, turnover
    fills: pd.DataFrame
    orders: pd.DataFrame
    positions: pd.DataFrame  # session_date, instrument_id, qty, price_local, value_base
    events: pd.DataFrame  # dividends, splits, borrow, fx conversions, rejections
    metrics: dict[str, Any]
    costs: dict[str, float]
    caveats: list[str]
    variant_count: int = 1
    benchmark: pd.Series | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def equity_series(self) -> pd.Series:
        return pd.Series(
            self.equity["equity"].to_numpy(),
            index=pd.DatetimeIndex(self.equity["session_date"]),
            name="equity",
        )

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"{self.strategy.get('name')} {m.get('start')}..{m.get('end')} "
            f"base {self.config.get('base_currency')}",
            f"  total return {m.get('total_return', float('nan')):+.2%}  "
            f"CAGR {m.get('cagr', float('nan')):+.2%}  "
            f"vol {m.get('volatility', float('nan')):.2%}  Sharpe {m.get('sharpe', float('nan')):.2f}  "
            f"max DD {m.get('max_drawdown', float('nan')):.2%}",
            f"  fills {len(self.fills)}  costs {sum(self.costs.values()):,.2f} "
            f"{self.config.get('base_currency')}  "
            f"variants tried in this process: {self.variant_count}",
            "  caveats:",
        ]
        lines += [f"    - {c}" for c in self.caveats]
        return "\n".join(lines)

    def to_dir(self, path: Path | str) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.equity.to_parquet(out / "equity.parquet", index=False)
        self.fills.to_parquet(out / "fills.parquet", index=False)
        self.orders.to_parquet(out / "orders.parquet", index=False)
        self.positions.to_parquet(out / "positions.parquet", index=False)
        self.events.to_parquet(out / "events.parquet", index=False)
        meta = {
            "strategy": self.strategy,
            "config": self.config,
            "metrics": self.metrics,
            "costs": self.costs,
            "caveats": self.caveats,
            "variant_count": self.variant_count,
        }
        (out / "result.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return out
