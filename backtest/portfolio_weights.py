"""Portfolio weight helpers shared by ranking simulation and concentration reporting."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def weights_for_symbols(symbols: list[str], *, gross: float = 1.0) -> dict[str, float]:
    if not symbols:
        return {}
    weight = gross / len(symbols)
    return {symbol: weight for symbol in symbols}


def capped_equal_weights(
    symbols: list[str],
    *,
    gross: float = 1.0,
    max_weight: float | None = None,
) -> dict[str, float]:
    """Equal-weight portfolio with optional per-name cap and redistribution."""
    if not symbols:
        return {}
    n = len(symbols)
    base_weight = gross / n
    if max_weight is None or max_weight >= base_weight:
        return {symbol: base_weight for symbol in symbols}

    weights = {symbol: min(base_weight, max_weight) for symbol in symbols}
    remaining = gross - sum(weights.values())
    adjustable = [symbol for symbol in symbols if weights[symbol] < max_weight - 1e-12]

    while remaining > 1e-12 and adjustable:
        increment = remaining / len(adjustable)
        next_adjustable: list[str] = []
        for symbol in adjustable:
            room = max_weight - weights[symbol]
            delta = min(increment, room)
            weights[symbol] += delta
            remaining -= delta
            if weights[symbol] < max_weight - 1e-12:
                next_adjustable.append(symbol)
        adjustable = next_adjustable

    total = sum(weights.values())
    if total <= 0:
        return weights_for_symbols(symbols, gross=gross)
    if total <= gross + 1e-12:
        return weights
    scale = gross / total
    return {symbol: weight * scale for symbol, weight in weights.items()}


def weighted_portfolio_return(
    cross_section: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    return_col: str,
) -> float:
    """Compute weighted portfolio return for one cross-section."""
    if not weights:
        return 0.0
    indexed = cross_section.set_index("symbol")
    total = 0.0
    for symbol, weight in weights.items():
        if symbol not in indexed.index:
            continue
        total += float(weight) * float(indexed.loc[symbol, return_col])
    return total
