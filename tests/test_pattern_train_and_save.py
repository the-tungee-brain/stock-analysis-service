"""Tests for model artifact training and saving."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.store import OHLCV_COLUMNS, save_raw
from features.build_features import build_and_save_features
from models.artifact_store import load_model_artifacts, meta_path, model_path
from models.labels import LabelScheme
from models.train_and_save import TrainAndSaveConfig, train_and_save
from models.xgb_model import XGBModelConfig
from tests.conftest import seed_pattern_benchmarks


def _synthetic_ohlcv(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=rows, freq="B", name="date")
    close = np.maximum(100 + np.cumsum(rng.normal(0, 0.5, size=rows)), 1.0)
    spread = rng.uniform(0.5, 2.0, size=rows)
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, size=rows),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=rows),
        },
        index=index,
    ).loc[:, list(OHLCV_COLUMNS)]


@pytest.fixture
def seeded_symbol_data(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)

    seed_pattern_benchmarks(rows=600)
    save_raw(_synthetic_ohlcv(rows=600), "AAPL")
    build_and_save_features("AAPL")

    return {
        "artifact_dir": artifact_dir,
        "train_end": pd.Timestamp("2021-12-31"),
    }


def test_train_and_save_writes_model_and_metadata(seeded_symbol_data):
    artifact_dir = seeded_symbol_data["artifact_dir"]
    config = TrainAndSaveConfig(
        symbols=("AAPL",),
        train_end_date=seeded_symbol_data["train_end"],
        artifact_dir=artifact_dir,
        model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
    )

    result = train_and_save(config)

    assert model_path(artifact_dir).exists()
    assert meta_path(artifact_dir).exists()
    assert result["n_rows"] >= 100
    assert result["n_features"] > 0
    assert result["train_end_date"] <= "2021-12-31"

    model, metadata = load_model_artifacts(artifact_dir)
    assert metadata["train_end_date"] == result["train_end_date"]
    assert metadata["feature_columns"]
    assert metadata["class_labels"] == [-1, 0, 1]
    assert metadata["class_mapping"] == {"-1": 0, "0": 1, "1": 2}
    assert model is not None

    meta_json = json.loads(meta_path(artifact_dir).read_text(encoding="utf-8"))
    assert meta_json["symbols"] == ["AAPL"]


def test_train_and_save_binary_metadata(seeded_symbol_data):
    artifact_dir = seeded_symbol_data["artifact_dir"]
    config = TrainAndSaveConfig(
        symbols=("AAPL",),
        train_end_date=seeded_symbol_data["train_end"],
        artifact_dir=artifact_dir,
        model_config=XGBModelConfig(
            n_estimators=10,
            max_depth=2,
            random_state=0,
            label_scheme=LabelScheme.BINARY_UPDOWN,
            use_class_weights=True,
        ),
        min_up_prob=0.65,
        universe="tradeable_v1",
    )

    train_and_save(config)
    _, metadata = load_model_artifacts(artifact_dir)

    assert metadata["label_scheme"] == "binary_updown"
    assert metadata["use_class_weights"] is True
    assert metadata["min_up_prob"] == 0.65
    assert metadata["universe"] == "tradeable_v1"
    assert metadata["class_labels"] == [0, 1]
