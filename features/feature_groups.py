"""Feature family definitions for ablation and simplicity tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

RELATIVE_STRENGTH_PREFIX = "rs_vs_spy_"
MOMENTUM_COLUMNS: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_21d",
    "ret_63d",
    "ret_126d",
    "ret_252d",
)
TREND_COLUMNS: tuple[str, ...] = (
    "close_vs_sma20",
    "close_vs_sma200",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
)
VOLUME_COLUMNS: tuple[str, ...] = (
    "vol_ratio_20d",
    "vol_zscore_20d",
    "vol_trend_5d_20d",
    "vol_chg_5d",
)
MARKET_CONTEXT_PREFIXES: tuple[str, ...] = ("spy_", "vix_")
TECHNICAL_COLUMNS: tuple[str, ...] = (
    "rsi_14",
    "macd",
    "macd_hist",
    "macd_signal",
    "bb_lower",
    "bb_mid",
    "bb_upper",
    "bb_pct",
    "atr_14",
)
VOLATILITY_COLUMNS: tuple[str, ...] = ("vol_20d",)

FEATURE_GROUP_ORDER: tuple[str, ...] = (
    "relative_strength",
    "momentum",
    "trend",
    "volume",
    "market_context",
)


@dataclass(frozen=True)
class FeatureGroupSpec:
    key: str
    label: str
    columns: tuple[str, ...]


def _match_prefix(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def classify_feature(name: str) -> str | None:
    """Assign a feature column to one family, or None if unclassified."""
    if name.startswith(RELATIVE_STRENGTH_PREFIX):
        return "relative_strength"
    if name in MOMENTUM_COLUMNS:
        return "momentum"
    if name in TREND_COLUMNS:
        return "trend"
    if name in VOLUME_COLUMNS:
        return "volume"
    if _match_prefix(name, MARKET_CONTEXT_PREFIXES):
        return "market_context"
    if name in TECHNICAL_COLUMNS:
        return "technical"
    if name in VOLATILITY_COLUMNS:
        return "momentum"
    return None


def group_features(feature_columns: Iterable[str]) -> dict[str, list[str]]:
    """Bucket available feature columns into families."""
    grouped: dict[str, list[str]] = {key: [] for key in FEATURE_GROUP_ORDER}
    grouped["technical"] = []
    grouped["other"] = []
    for column in feature_columns:
        family = classify_feature(column)
        if family is None:
            grouped["other"].append(column)
        else:
            grouped[family].append(column)
    return grouped


def feature_group_specs(feature_columns: list[str]) -> list[FeatureGroupSpec]:
    """Return non-empty ablation groups present in the panel."""
    grouped = group_features(feature_columns)
    labels = {
        "relative_strength": "Relative strength vs SPY",
        "momentum": "Momentum",
        "trend": "Trend / moving averages",
        "volume": "Volume",
        "market_context": "Market context (SPY/VIX)",
    }
    specs: list[FeatureGroupSpec] = []
    for key in FEATURE_GROUP_ORDER:
        columns = grouped.get(key, [])
        if columns:
            specs.append(FeatureGroupSpec(key=key, label=labels[key], columns=tuple(columns)))
    return specs


def simplicity_feature_columns(feature_columns: list[str]) -> list[str]:
    """Benchmark model: relative strength + momentum + market context only."""
    allowed = set(FEATURE_GROUP_ORDER) - {"trend", "volume"}
    grouped = group_features(feature_columns)
    selected: list[str] = []
    for key in allowed:
        selected.extend(grouped.get(key, []))
    return sorted(selected)


def without_group(feature_columns: list[str], group_key: str) -> list[str]:
    """All features except one family."""
    grouped = group_features(feature_columns)
    excluded = set(grouped.get(group_key, []))
    return [column for column in feature_columns if column not in excluded]


def only_group(feature_columns: list[str], group_key: str) -> list[str]:
    """Features from a single family."""
    grouped = group_features(feature_columns)
    return list(grouped.get(group_key, []))


def combine_groups(feature_columns: list[str], group_keys: Iterable[str]) -> list[str]:
    """Union of multiple feature families."""
    grouped = group_features(feature_columns)
    selected: list[str] = []
    for key in group_keys:
        selected.extend(grouped.get(key, []))
    return sorted(dict.fromkeys(selected))


PHASE5_MODELS: tuple[tuple[str, str, tuple[str, ...] | None], ...] = (
    ("A", "Relative strength only", ("relative_strength",)),
    ("B", "Trend only", ("trend",)),
    ("C", "Relative strength + trend", ("relative_strength", "trend")),
    ("D", "Relative strength + trend + market context", ("relative_strength", "trend", "market_context")),
    ("E", "Simple benchmark (RS + momentum + market context)", ("simple",)),
    ("F", "Full model", None),
)


def phase5_model_features(feature_columns: list[str], model_key: str) -> list[str]:
    """Resolve feature columns for Phase 5 minimal model variants."""
    key = model_key.strip().upper()
    for spec_key, _, groups in PHASE5_MODELS:
        if spec_key != key:
            continue
        if groups is None:
            return list(feature_columns)
        if groups == ("simple",):
            return simplicity_feature_columns(feature_columns)
        return combine_groups(feature_columns, groups)
    raise ValueError(f"Unknown Phase 5 model key: {model_key}")


def phase5_model_label(model_key: str) -> str:
    key = model_key.strip().upper()
    for spec_key, label, _ in PHASE5_MODELS:
        if spec_key == key:
            return label
    raise ValueError(f"Unknown Phase 5 model key: {model_key}")
