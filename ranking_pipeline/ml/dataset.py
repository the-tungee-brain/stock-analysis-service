"""Training dataset assembly for ranking ML models."""

from __future__ import annotations

import pandas as pd

from models.labels import EXCESS_RETURN_COLUMN, EXCLUDE_FROM_FEATURES
from ranking_pipeline.ml.labels import (
    TOP_QUINTILE_LABEL_COLUMN,
    ClassificationTarget,
    add_top_quintile_labels,
    classification_column,
    regression_column,
)
from ranking_pipeline.features.parquet_store import load_ranking_features
from ranking_pipeline.features.ranking_features import all_ranking_feature_columns


def feature_columns_for_ml(df: pd.DataFrame) -> list[str]:
    """Numeric columns suitable for ML (excludes forward-looking labels)."""
    preferred = [c for c in all_ranking_feature_columns() if c in df.columns]
    numeric = df.select_dtypes(include="number").columns
    exclude = set(EXCLUDE_FROM_FEATURES) | {TOP_QUINTILE_LABEL_COLUMN}
    extra = [
        c
        for c in numeric
        if c not in exclude
        and not c.startswith("label_")
        and c not in preferred
    ]
    return preferred + [c for c in extra if c.startswith("pat_")]


def build_panel_from_symbols(
    symbols: list[str],
    *,
    end_date: pd.Timestamp | None = None,
    classification_target: ClassificationTarget = ClassificationTarget.OUTPERFORM_SPY,
) -> pd.DataFrame:
    """Stack per-symbol feature history into a multi-index panel (symbol, date)."""
    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            df = load_ranking_features(symbol)
        except FileNotFoundError:
            continue
        if end_date is not None:
            df = df[df.index <= end_date]
        if df.empty:
            continue
        chunk = df.copy()
        chunk["symbol"] = symbol.strip().upper()
        frames.append(chunk.reset_index().set_index(["symbol", "date"]))
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=0).sort_index()
    if classification_target == ClassificationTarget.TOP_QUINTILE:
        panel = add_top_quintile_labels(panel)
    return panel


def train_test_split_by_date(
    panel: pd.DataFrame,
    *,
    train_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = panel.index.get_level_values("date")
    cutoff = pd.Timestamp(train_end)
    train = panel[dates <= cutoff]
    test = panel[dates > cutoff]
    return train, test


def xy_from_panel(
    panel: pd.DataFrame,
    feature_cols: list[str] | None = None,
    *,
    classification_target: ClassificationTarget = ClassificationTarget.OUTPERFORM_SPY,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = feature_cols or feature_columns_for_ml(panel)
    X = panel[cols].astype("float64").fillna(0.0)
    cls_col = classification_column(classification_target)
    reg_col = regression_column()
    y_cls = panel[cls_col].astype(int)
    y_reg = panel[reg_col].astype("float64")
    valid = y_cls.notna() & y_reg.notna()
    return X.loc[valid], y_cls.loc[valid], y_reg.loc[valid]
