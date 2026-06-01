"""Shared prediction logic for standalone and main FastAPI apps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.download import download_symbol
from data.loader import load_symbol
from data.store import save_raw
from features.build_features import build_features, features_ready_slice
from models.artifact_store import (
    load_model_artifacts,
    metadata_class_labels,
    metadata_label_scheme,
    resolve_artifact_dir,
)
from models.xgb_model import predict_xgb

KEY_INDICATORS: tuple[str, ...] = (
    "rsi_14",
    "sma_20",
    "sma_200",
    "macd",
    "bb_pct",
)


@dataclass(frozen=True)
class LoadedModel:
    model: object
    metadata: dict[str, Any]
    feature_columns: list[str]


def load_deployed_model(artifact_dir: Path | str | None = None) -> LoadedModel:
    model, metadata = load_model_artifacts(artifact_dir)
    feature_columns = list(metadata["feature_columns"])
    return LoadedModel(model=model, metadata=metadata, feature_columns=feature_columns)


def ensure_raw_ohlcv(symbol: str) -> pd.DataFrame:
    """Load raw OHLCV from Parquet, downloading via yfinance when missing."""
    symbol_upper = symbol.strip().upper()
    try:
        return load_symbol(symbol_upper)
    except FileNotFoundError:
        raw = download_symbol(symbol_upper)
        save_raw(raw, symbol_upper)
        return raw


def build_latest_feature_row(symbol: str) -> pd.Series:
    """Rebuild features from raw OHLCV and return the latest ready row."""
    raw = ensure_raw_ohlcv(symbol)
    features = build_features(raw)
    ready = features_ready_slice(features)
    if ready.empty:
        raise ValueError(f"Insufficient history to build features for {symbol.strip().upper()}")
    return ready.iloc[-1]


def _extract_up_prob(
    probabilities: dict[str, float],
    class_labels: tuple[int, ...],
) -> float | None:
    if 1 in class_labels:
        return probabilities.get("1")
    if -1 in class_labels and 1 in class_labels:
        return probabilities.get("1")
    return None


def predict_for_symbol(symbol: str, loaded: LoadedModel) -> dict[str, Any]:
    symbol_upper = symbol.strip().upper()
    latest = build_latest_feature_row(symbol_upper)
    metadata = loaded.metadata

    missing = [col for col in loaded.feature_columns if col not in latest.index]
    if missing:
        raise ValueError(f"Missing feature columns for {symbol_upper}: {missing}")

    label_scheme = metadata_label_scheme(metadata)
    class_labels = metadata_class_labels(metadata)
    min_up_prob = metadata.get("min_up_prob")
    training_symbols = {str(item).upper() for item in metadata.get("symbols", [])}

    feature_row = latest[loaded.feature_columns].to_frame().T
    y_pred, y_proba = predict_xgb(
        loaded.model,
        feature_row,
        label_scheme=label_scheme,
    )
    prediction = int(y_pred[0])

    probabilities = {
        str(label): float(y_proba[0, idx])
        for idx, label in enumerate(class_labels)
    }
    up_prob = _extract_up_prob(probabilities, class_labels)
    trade_signal = None
    if min_up_prob is not None and up_prob is not None:
        trade_signal = float(up_prob) >= float(min_up_prob)

    indicators = {
        name: float(latest[name])
        for name in KEY_INDICATORS
        if name in latest.index and pd.notna(latest[name])
    }

    return {
        "symbol": symbol_upper,
        "date": pd.Timestamp(latest.name).strftime("%Y-%m-%d"),
        "label_scheme": label_scheme.value,
        "prediction": prediction,
        "probabilities": probabilities,
        "up_prob": up_prob,
        "trade_signal": trade_signal,
        "min_up_prob": min_up_prob,
        "in_training_universe": symbol_upper in training_symbols,
        "model_train_end_date": metadata.get("train_end_date"),
        "model_universe": metadata.get("universe"),
        "indicators": indicators,
    }


def health_payload(loaded: LoadedModel) -> dict[str, Any]:
    metadata = loaded.metadata
    return {
        "status": "ok",
        "model": {
            "train_end_date": metadata.get("train_end_date"),
            "train_start_date": metadata.get("train_start_date"),
            "n_features": metadata.get("n_features"),
            "symbols": metadata.get("symbols", []),
            "label_scheme": metadata.get("label_scheme"),
            "use_class_weights": metadata.get("use_class_weights"),
            "min_up_prob": metadata.get("min_up_prob"),
            "universe": metadata.get("universe"),
        },
    }
