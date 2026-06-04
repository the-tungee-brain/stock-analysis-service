"""Identify feature ranges with highest and lowest expectancy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.models import FeatureConditionInsight, FeatureSnapshot

_MIN_BIN_TRADES = 3
_TOP_N = 5

_FEATURE_EXTRACTORS: dict[str, Callable[[FeatureSnapshot], float | None]] = {
    "rs_percentile": lambda s: s.rs_percentile,
    "volume_ratio": lambda s: s.volume_ratio,
    "close_vs_sma50": lambda s: s.close_vs_sma50,
    "close_vs_sma200": lambda s: s.close_vs_sma200,
    "distance_to_20d_high": lambda s: s.distance_to_20d_high,
}


@dataclass(frozen=True, slots=True)
class _FeatureBin:
    feature: str
    bin_start: float
    bin_end: float
    trades: tuple[HistoricalTrade, ...]

    @property
    def expectancy(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.return_pct for t in self.trades) / len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.return_pct > 0)
        return wins / len(self.trades)


def _quintile_edges(values: list[float]) -> list[float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    edges = [sorted_vals[0]]
    for q in range(1, 5):
        idx = min(n - 1, int((q / 5.0) * n))
        edges.append(sorted_vals[idx])
    edges.append(sorted_vals[-1])
    return edges


def _assign_bin(value: float, edges: list[float]) -> tuple[float, float]:
    for idx in range(len(edges) - 1):
        start = edges[idx]
        end = edges[idx + 1]
        is_last = idx == len(edges) - 2
        if start <= value <= end or (is_last and value >= start):
            return start, end
    return edges[0], edges[-1]


def _bins_for_feature(
    trades: Sequence[HistoricalTrade],
    feature: str,
) -> list[_FeatureBin]:
    extractor = _FEATURE_EXTRACTORS[feature]
    labeled: list[tuple[float, HistoricalTrade]] = []
    for trade in trades:
        if trade.feature_snapshot is None:
            continue
        raw = extractor(trade.feature_snapshot)
        if raw is None:
            continue
        labeled.append((raw, trade))

    if len(labeled) < _MIN_BIN_TRADES:
        return []

    values = [value for value, _ in labeled]
    edges = _quintile_edges(values)
    buckets: dict[tuple[float, float], list[HistoricalTrade]] = {}
    for value, trade in labeled:
        key = _assign_bin(value, edges)
        buckets.setdefault(key, []).append(trade)

    bins: list[_FeatureBin] = []
    for (start, end), bucket in sorted(buckets.items()):
        if len(bucket) < _MIN_BIN_TRADES:
            continue
        bins.append(
            _FeatureBin(
                feature=feature,
                bin_start=start,
                bin_end=end,
                trades=tuple(bucket),
            )
        )
    return bins


def _to_insight(bin_row: _FeatureBin) -> FeatureConditionInsight:
    return FeatureConditionInsight(
        feature=bin_row.feature,
        range_label=f"{bin_row.bin_start:.4g} – {bin_row.bin_end:.4g}",
        bin_start=bin_row.bin_start,
        bin_end=bin_row.bin_end,
        trade_count=len(bin_row.trades),
        expectancy=round(bin_row.expectancy, 6),
        win_rate=round(bin_row.win_rate, 4),
    )


def analyze_feature_conditions(
    trades: Sequence[HistoricalTrade],
    *,
    top_n: int = _TOP_N,
) -> tuple[tuple[FeatureConditionInsight, ...], tuple[FeatureConditionInsight, ...]]:
    all_bins: list[_FeatureBin] = []
    for feature in _FEATURE_EXTRACTORS:
        all_bins.extend(_bins_for_feature(trades, feature))

    if not all_bins:
        return (), ()

    ranked = sorted(all_bins, key=lambda row: row.expectancy, reverse=True)
    top = tuple(_to_insight(row) for row in ranked[:top_n])
    worst = tuple(_to_insight(row) for row in ranked[-top_n:][::-1])
    return top, worst
