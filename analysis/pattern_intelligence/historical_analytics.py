"""Historical performance stats for detected candlestick patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import PATTERN_CATALOG, scan_candlestick_patterns
from analysis.pattern_intelligence.trend_context import TrendContext


@dataclass(frozen=True)
class PatternHistoricalStats:
    pattern_id: str
    label: str
    occurrence_count: int
    avg_return_5d: float | None
    avg_return_20d: float | None
    win_rate_5d: float | None
    win_rate_20d: float | None
    max_drawdown_20d: float | None


@dataclass(frozen=True)
class SetupOutcomeStats:
    """Historical outcomes when pattern, trend, and RS context align with today."""

    label: str
    pattern_label: str
    trend_label: str
    rs_label: str
    occurrence_count: int
    pattern_only_count: int
    avg_return_5d: float | None
    avg_return_20d: float | None
    win_rate_5d: float | None
    win_rate_20d: float | None
    max_drawdown_20d: float | None


def compute_setup_outcome_stats(
    ohlcv: pd.DataFrame,
    *,
    pattern_id: str,
    pattern_label: str,
    context: TrendContext,
    features: pd.DataFrame | None = None,
    min_occurrences: int = 3,
    analytics_years: int = 5,
) -> SetupOutcomeStats | None:
    """Filter pattern hits by current trend + RS regime for combined setup stats."""
    if pattern_id not in PATTERN_CATALOG or len(ohlcv) < 30:
        return None

    trend_label, rs_label = _setup_context_labels(context)
    label = f"{pattern_label} · {trend_label} · {rs_label}"

    close = ohlcv["close"].astype(float)
    if analytics_years > 0:
        cutoff = close.index.max() - pd.DateOffset(years=analytics_years)
        ohlcv = ohlcv.loc[ohlcv.index >= cutoff]
        close = close.loc[close.index >= cutoff]

    scan = scan_candlestick_patterns(ohlcv)
    hit_col = f"hit_{pattern_id}"
    if hit_col not in scan.columns:
        return None

    pattern_only_idx = scan.index[scan[hit_col].astype(bool)]
    pattern_only_count = int(len(pattern_only_idx))

    if features is None:
        from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, ensure_benchmark_ohlcv, load_benchmark_ohlcv
        from features.build_features import build_features
        from features.market_context import attach_market_context

        ensure_benchmark_ohlcv()
        features = build_features(ohlcv)
        spy_close = load_benchmark_ohlcv(BENCHMARK_SYMBOL)["close"]
        vix_close = load_benchmark_ohlcv(VIX_SYMBOL)["close"]
        features = attach_market_context(
            features,
            stock_close=ohlcv["close"],
            spy_close=spy_close,
            vix_close=vix_close,
        )

    filtered_idx = [
        dt
        for dt in pattern_only_idx
        if _matches_setup_context(features.loc[dt] if dt in features.index else None, context)
    ]

    stats = _forward_stats(close, filtered_idx)
    if stats is None:
        return SetupOutcomeStats(
            label=label,
            pattern_label=pattern_label,
            trend_label=trend_label,
            rs_label=rs_label,
            occurrence_count=len(filtered_idx),
            pattern_only_count=pattern_only_count,
            avg_return_5d=None,
            avg_return_20d=None,
            win_rate_5d=None,
            win_rate_20d=None,
            max_drawdown_20d=None,
        )

    rets_5d, rets_20d, drawdowns_20d = stats
    return SetupOutcomeStats(
        label=label,
        pattern_label=pattern_label,
        trend_label=trend_label,
        rs_label=rs_label,
        occurrence_count=len(filtered_idx),
        pattern_only_count=pattern_only_count,
        avg_return_5d=_mean(rets_5d),
        avg_return_20d=_mean(rets_20d),
        win_rate_5d=_win_rate(rets_5d),
        win_rate_20d=_win_rate(rets_20d),
        max_drawdown_20d=_mean(drawdowns_20d),
    )


def _setup_context_labels(context: TrendContext) -> tuple[str, str]:
    if context.above_sma_200 is True:
        trend_label = "Above SMA200"
    elif context.above_sma_200 is False:
        trend_label = "Below SMA200"
    else:
        trend_label = context.trend_bias.replace("_", " ").title()

    rs = context.rs_vs_spy_63d
    if rs is None:
        rs = context.rs_vs_spy_21d
    if rs is None:
        rs_label = "RS vs SPY n/a"
    elif rs > 0:
        rs_label = "RS leading SPY"
    else:
        rs_label = "RS lagging SPY"
    return trend_label, rs_label


def _matches_setup_context(row: pd.Series | None, context: TrendContext) -> bool:
    if row is None:
        return False

    if context.above_sma_200 is not None:
        vs_sma200 = row.get("close_vs_sma200")
        if pd.isna(vs_sma200):
            sma_200 = row.get("sma_200")
            close = row.get("close")
            if pd.isna(sma_200) or pd.isna(close):
                return False
            above = float(close) > float(sma_200)
        else:
            above = float(vs_sma200) > 0
        if above != context.above_sma_200:
            return False

    rs_now = context.rs_vs_spy_63d
    if rs_now is None:
        rs_now = context.rs_vs_spy_21d
    if rs_now is not None:
        rs_hist = row.get("rs_vs_spy_63d")
        if pd.isna(rs_hist):
            rs_hist = row.get("rs_vs_spy_21d")
        if pd.isna(rs_hist):
            return False
        if (float(rs_hist) > 0) != (float(rs_now) > 0):
            return False

    return True


def _forward_stats(
    close: pd.Series,
    hit_idx: pd.Index | list,
) -> tuple[list[float], list[float], list[float]] | None:
    rets_5d: list[float] = []
    rets_20d: list[float] = []
    drawdowns_20d: list[float] = []

    for dt in hit_idx:
        loc = close.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.start
        if int(loc) + 20 >= len(close):
            continue
        entry = float(close.iloc[int(loc)])
        if entry <= 0:
            continue
        path = close.iloc[int(loc) : int(loc) + 21]
        ret_5 = (
            float(close.iloc[int(loc) + 5] / entry - 1.0)
            if int(loc) + 5 < len(close)
            else None
        )
        ret_20 = float(close.iloc[int(loc) + 20] / entry - 1.0)
        rets_20d.append(ret_20)
        if ret_5 is not None:
            rets_5d.append(ret_5)
        running_max = path.cummax()
        dd = (path / running_max - 1.0).min()
        drawdowns_20d.append(float(dd))

    if not rets_20d:
        return None
    return rets_5d, rets_20d, drawdowns_20d


def compute_pattern_historical_stats(
    ohlcv: pd.DataFrame,
    *,
    pattern_id: str,
    label: str,
    min_occurrences: int = 3,
    analytics_years: int = 5,
) -> PatternHistoricalStats | None:
    if pattern_id not in PATTERN_CATALOG or len(ohlcv) < 30:
        return None

    close = ohlcv["close"].astype(float)
    if analytics_years > 0:
        cutoff = close.index.max() - pd.DateOffset(years=analytics_years)
        ohlcv = ohlcv.loc[ohlcv.index >= cutoff]
        close = close.loc[close.index >= cutoff]

    scan = scan_candlestick_patterns(ohlcv)
    hit_col = f"hit_{pattern_id}"
    if hit_col not in scan.columns:
        return None

    hit_idx = scan.index[scan[hit_col].astype(bool)]
    if len(hit_idx) < min_occurrences:
        return PatternHistoricalStats(
            pattern_id=pattern_id,
            label=label,
            occurrence_count=int(len(hit_idx)),
            avg_return_5d=None,
            avg_return_20d=None,
            win_rate_5d=None,
            win_rate_20d=None,
            max_drawdown_20d=None,
        )

    stats = _forward_stats(close, hit_idx)
    if stats is None:
        return PatternHistoricalStats(
            pattern_id=pattern_id,
            label=label,
            occurrence_count=int(len(hit_idx)),
            avg_return_5d=None,
            avg_return_20d=None,
            win_rate_5d=None,
            win_rate_20d=None,
            max_drawdown_20d=None,
        )
    rets_5d, rets_20d, drawdowns_20d = stats

    return PatternHistoricalStats(
        pattern_id=pattern_id,
        label=label,
        occurrence_count=int(len(hit_idx)),
        avg_return_5d=_mean(rets_5d),
        avg_return_20d=_mean(rets_20d),
        win_rate_5d=_win_rate(rets_5d),
        win_rate_20d=_win_rate(rets_20d),
        max_drawdown_20d=_mean(drawdowns_20d),
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _win_rate(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean([1.0 if v > 0 else 0.0 for v in values]))
