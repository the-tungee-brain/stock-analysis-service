"""Position sizing from ranked candidates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ranking_pipeline.portfolio.config import SizingMode


@dataclass(frozen=True)
class RankedCandidate:
    symbol: str
    final_score: float
    expected_excess_return: float | None = None
    ml_probability: float | None = None
    atr_14: float | None = None


def _normalize_weights(raw: pd.Series) -> pd.Series:
    total = raw.sum()
    if total <= 0 or pd.isna(total):
        n = len(raw)
        return pd.Series(1.0 / n, index=raw.index) if n else raw
    return raw / total


def equal_weight(candidates: list[RankedCandidate]) -> pd.Series:
    symbols = [c.symbol for c in candidates]
    if not symbols:
        return pd.Series(dtype="float64")
    w = 1.0 / len(symbols)
    return pd.Series(w, index=symbols)


def score_weighted(candidates: list[RankedCandidate]) -> pd.Series:
    scores = pd.Series(
        {c.symbol: max(c.final_score, 1e-6) for c in candidates},
        dtype="float64",
    )
    return _normalize_weights(scores)


def volatility_adjusted(candidates: list[RankedCandidate]) -> pd.Series:
    raw: dict[str, float] = {}
    for c in candidates:
        atr = c.atr_14
        if atr is None or atr <= 0 or pd.isna(atr):
            atr = 1.0
        raw[c.symbol] = max(c.final_score, 1e-6) / float(atr)
    return _normalize_weights(pd.Series(raw, dtype="float64"))


def compute_target_weights(
    candidates: list[RankedCandidate],
    mode: SizingMode,
) -> pd.Series:
    """Return target weights summing to 1."""
    if not candidates:
        return pd.Series(dtype="float64")
    if mode == SizingMode.EQUAL_WEIGHT:
        return equal_weight(candidates)
    if mode == SizingMode.SCORE_WEIGHTED:
        return score_weighted(candidates)
    if mode == SizingMode.VOLATILITY_ADJUSTED:
        return volatility_adjusted(candidates)
    raise ValueError(f"Unknown sizing mode: {mode}")
