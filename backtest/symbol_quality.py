"""Per-symbol backtest quality filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SymbolQualityConfig:
    """Minimum per-symbol stats required to mark a symbol as tradeable."""

    min_trades: int = 50
    min_pf: float = 1.3
    min_sharpe: float = 1.0

    def __post_init__(self) -> None:
        if self.min_trades < 0:
            raise ValueError("min_trades must be non-negative")
        if self.min_pf < 0:
            raise ValueError("min_pf must be non-negative")


def default_symbol_quality() -> SymbolQualityConfig:
    return SymbolQualityConfig()


def filter_recommended_symbols(
    per_symbol_stats: list[dict[str, Any]],
    criteria: SymbolQualityConfig | None = None,
) -> list[dict[str, Any]]:
    """Return per-symbol rows that satisfy all quality thresholds."""
    cfg = criteria or default_symbol_quality()
    recommended: list[dict[str, Any]] = []

    for row in per_symbol_stats:
        n_trades = int(row.get("n_trades", 0))
        if n_trades < cfg.min_trades:
            continue

        pf = float(row.get("profit_factor", float("nan")))
        sharpe = float(row.get("sharpe_ratio", float("nan")))
        if np.isnan(pf) or pf < cfg.min_pf:
            continue
        if np.isnan(sharpe) or sharpe < cfg.min_sharpe:
            continue

        recommended.append(row)

    return sorted(recommended, key=lambda item: str(item["symbol"]))
