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
from models.artifact_store import load_model_artifacts, resolve_artifact_dir
from models.xgb_model import MODEL_CLASS_LABELS, predict_xgb

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


def predict_for_symbol(symbol: str, loaded: LoadedModel) -> dict[str, Any]:
    symbol_upper = symbol.strip().upper()
    latest = build_latest_feature_row(symbol_upper)

    missing = [col for col in loaded.feature_columns if col not in latest.index]
    if missing:
        raise ValueError(f"Missing feature columns for {symbol_upper}: {missing}")

    feature_row = latest[loaded.feature_columns].to_frame().T
    y_pred, y_proba = predict_xgb(loaded.model, feature_row)
    prediction = int(y_pred[0])

    probabilities = {
        str(label): float(y_proba[0, idx])
        for idx, label in enumerate(MODEL_CLASS_LABELS)
    }
    indicators = {
        name: float(latest[name])
        for name in KEY_INDICATORS
        if name in latest.index and pd.notna(latest[name])
    }

    return {
        "symbol": symbol_upper,
        "date": pd.Timestamp(latest.name).strftime("%Y-%m-%d"),
        "prediction": prediction,
        "probabilities": probabilities,
        "indicators": indicators,
    }


def health_payload(loaded: LoadedModel) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": {
            "train_end_date": loaded.metadata.get("train_end_date"),
            "train_start_date": loaded.metadata.get("train_start_date"),
            "n_features": loaded.metadata.get("n_features"),
            "symbols": loaded.metadata.get("symbols", []),
        },
    }
