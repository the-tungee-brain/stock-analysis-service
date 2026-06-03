"""Weighted composite score and per-group contribution breakdown."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.features.ranking_features import FEATURE_GROUPS, PATTERN_SCORE_COLUMNS
from ranking_pipeline.scoring.normalization import zscore_cross_section

# Higher breakout distance (near highs) is bullish; invert distance columns for scoring
BREAKOUT_INVERT = frozenset({"dist_20d_high", "dist_52w_high"})

GROUP_FEATURE_MAP: dict[str, list[str]] = {
    **FEATURE_GROUPS,
    "pattern": list(PATTERN_SCORE_COLUMNS) + ["pattern_signal_score"],
}
# Volatility used in features but folded into trend-adjacent risk; keep in trend extension optional
# For composite weights user spec, volatility not separate — atr features inform via breakout/vol not weighted separately
# We map volatility columns into breakout group for scoring purposes
GROUP_FEATURE_MAP["breakout"] = list(FEATURE_GROUPS["breakout"]) + list(FEATURE_GROUPS["volatility"])


@dataclass(frozen=True)
class CompositeScoreResult:
    symbol: str
    composite_score: float
    contributions: dict[str, float]


def score_universe_slice(
    panel: pd.DataFrame,
    config: RankingPipelineConfig,
) -> list[CompositeScoreResult]:
    """
    Score one cross-section (index = symbol, columns = raw features).

    ``panel`` must contain one row per symbol for the same as-of date.
    """
    weights = config.normalized_weights()
    zscored = zscore_cross_section(panel, [c for cols in GROUP_FEATURE_MAP.values() for c in cols])

    results: list[CompositeScoreResult] = []
    for symbol in panel.index:
        contributions: dict[str, float] = {}
        total = 0.0
        for group, weight in weights.items():
            cols = [c for c in GROUP_FEATURE_MAP.get(group, []) if c in zscored.columns]
            if not cols:
                continue
            row = zscored.loc[symbol, cols].astype("float64")
            for col in cols:
                if col in BREAKOUT_INVERT:
                    row[col] = -row[col]
            group_score = float(row.mean())
            contrib = weight * group_score
            contributions[group] = contrib
            total += contrib
        results.append(
            CompositeScoreResult(
                symbol=str(symbol),
                composite_score=total,
                contributions=contributions,
            )
        )
    return results


def composite_to_series(results: list[CompositeScoreResult]) -> pd.Series:
    return pd.Series(
        {r.symbol: r.composite_score for r in results},
        dtype="float64",
    ).sort_values(ascending=False)
