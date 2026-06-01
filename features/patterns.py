"""Candlestick pattern features via pandas-ta."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pandas_ta as ta

# Core patterns from architecture; TA-Lib unlocks the full set when installed.
PATTERN_NAMES: tuple[str, ...] = (
    "hammer",
    "doji",
    "engulfing",
    "morningstar",
    "shootingstar",
    "harami",
    "invertedhammer",
)

NATIVE_PATTERN_FUNCS: dict[str, str] = {
    "doji": "cdl_doji",
    "inside": "cdl_inside",
}


def _talib_available() -> bool:
    return importlib.util.find_spec("talib") is not None


def _normalize_pattern_column(name: str, series: pd.Series) -> pd.Series:
    values = series.fillna(0)
    # pandas-ta / TA-Lib use {-100, 0, 100}; store signed direction as float.
    return values.astype("float64").rename(f"pat_{name}")


def compute_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return binary/signed candlestick pattern columns prefixed with ``pat_``."""
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    columns: dict[str, pd.Series] = {}

    for pattern_name, func_name in NATIVE_PATTERN_FUNCS.items():
        func = getattr(ta, func_name)
        columns[f"pat_{pattern_name}"] = _normalize_pattern_column(
            pattern_name,
            func(open_, high, low, close),
        )

    talib_patterns = [name for name in PATTERN_NAMES if name not in NATIVE_PATTERN_FUNCS]
    if talib_patterns and _talib_available():
        detected = ta.cdl_pattern(open_, high, low, close, name=talib_patterns)
        if detected is not None and not detected.empty:
            for col in detected.columns:
                pattern_key = _pattern_name_from_column(col)
                columns[f"pat_{pattern_key}"] = _normalize_pattern_column(
                    pattern_key,
                    detected[col],
                )

    if not columns:
        return pd.DataFrame(index=df.index)

    return pd.DataFrame(columns, index=df.index)


def _pattern_name_from_column(column: str) -> str:
    name = column.upper()
    if name.startswith("CDL_"):
        name = name[4:]
    return name.lower().split("_", 1)[0]
