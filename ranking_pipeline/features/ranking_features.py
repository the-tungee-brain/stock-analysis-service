"""Ranking-specific feature matrix per symbol."""

from __future__ import annotations

import pandas as pd

from features.indicators import compute_indicators
from features.patterns import compute_patterns
from ranking_pipeline.features.decay import apply_feature_decay
from ranking_pipeline.validation.leakage import assert_no_feature_leakage
from models.labels import (
    BINARY_OUTPERFORM_SPY_COLUMN,
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    add_labels,
)

FEATURE_GROUPS: dict[str, list[str]] = {
    "trend": [
        "close_vs_sma20",
        "close_vs_sma50",
        "close_vs_sma200",
        "sma20_slope_5d",
        "sma50_slope_5d",
    ],
    "relative_strength": [
        "excess_ret_5d_vs_spy",
        "excess_ret_20d_vs_spy",
        "excess_ret_60d_vs_spy",
    ],
    "volume": [
        "rel_volume",
        "vol_ratio_20d",
    ],
    "breakout": [
        "dist_20d_high",
        "dist_52w_high",
        "new_high_20d",
        "new_high_52w",
    ],
    "volatility": [
        "atr_14",
        "atr_percentile_252d",
    ],
}

PATTERN_SCORE_COLUMNS = ("pat_engulfing", "pat_hammer", "pat_morningstar")


def all_ranking_feature_columns() -> list[str]:
    cols: list[str] = []
    for group_cols in FEATURE_GROUPS.values():
        cols.extend(group_cols)
    cols.extend(PATTERN_SCORE_COLUMNS)
    cols.append("pattern_signal_score")
    return cols


def compute_ranking_features(
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    *,
    include_labels: bool = True,
    apply_decay: bool = True,
    decay_halflife_days: float = 10.0,
    validate_leakage: bool = True,
) -> pd.DataFrame:
    """Build daily ranking features from OHLCV and SPY benchmark."""
    df = ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    indicators = compute_indicators(df)
    patterns = compute_patterns(df)

    out = pd.DataFrame(index=df.index)
    for pat_name in ("engulfing", "hammer", "morningstar"):
        col = f"pat_{pat_name}"
        if col in patterns.columns:
            out[col] = patterns[col].astype("float64")
    for length in (20, 50, 200):
        sma = indicators.get(f"sma_{length}")
        if sma is not None:
            out[f"close_vs_sma{length}"] = close / sma - 1.0

    sma20 = indicators.get("sma_20")
    sma50 = indicators.get("sma_50")
    if sma20 is not None:
        out["sma20_slope_5d"] = sma20.pct_change(5)
    if sma50 is not None:
        out["sma50_slope_5d"] = sma50.pct_change(5)

    spy = spy_close.reindex(out.index).astype("float64")
    for horizon, col in ((5, "excess_ret_5d_vs_spy"), (20, "excess_ret_20d_vs_spy"), (60, "excess_ret_60d_vs_spy")):
        stock_ret = close.pct_change(horizon)
        spy_ret = spy.pct_change(horizon)
        out[col] = stock_ret - spy_ret

    vol_ma20 = volume.rolling(20, min_periods=10).mean()
    out["vol_ratio_20d"] = volume / vol_ma20
    out["rel_volume"] = out["vol_ratio_20d"]

    # Shift(1): rolling extrema use only prior bars (no same-bar high leakage)
    high_prior = high.shift(1)
    high_20 = high_prior.rolling(20, min_periods=10).max()
    high_52w = high_prior.rolling(252, min_periods=60).max()
    out["dist_20d_high"] = close / high_20 - 1.0
    out["dist_52w_high"] = close / high_52w - 1.0
    out["new_high_20d"] = (close >= high_20).astype("float64")
    out["new_high_52w"] = (close >= high_52w).astype("float64")

    atr = indicators.get("atr_14")
    if atr is not None:
        out["atr_14"] = atr
        out["atr_percentile_252d"] = atr.rolling(252, min_periods=60).apply(
            lambda s: float(pd.Series(s).rank(pct=True).iloc[-1]) if len(s) else float("nan"),
            raw=False,
        )

    pattern_cols = [c for c in out.columns if c.startswith("pat_")]
    if pattern_cols:
        bullish = out[pattern_cols].clip(-100, 100) / 100.0
        out["pattern_signal_score"] = bullish.max(axis=1)
    else:
        out["pattern_signal_score"] = 0.0

    if include_labels:
        out = add_labels(out, close, benchmark_close=spy)
        out = out.dropna(subset=[FUTURE_RETURN_COLUMN])

    if apply_decay:
        out = apply_feature_decay(out, halflife_days=decay_halflife_days)

    feature_cols = [c for c in all_ranking_feature_columns() if c in out.columns]
    out = out.dropna(subset=feature_cols, how="any")

    if validate_leakage:
        assert_no_feature_leakage(out, df)

    return out
