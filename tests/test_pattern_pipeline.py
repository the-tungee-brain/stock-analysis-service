"""Tests for the end-to-end pattern training pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from models.train_pipeline import run_pipeline
from tests.test_pattern_train_and_save import _synthetic_ohlcv


@pytest.fixture
def pipeline_dirs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)
    monkeypatch.setattr("models.artifact_store.DEFAULT_ARTIFACT_DIR", artifact_dir)
    return artifact_dir


def test_run_pipeline_trains_model(pipeline_dirs):
    ohlcv = _synthetic_ohlcv(rows=500)

    with patch("models.train_pipeline.download_and_store_all") as download_mock:
        download_mock.return_value = {"AAPL": ohlcv}
        with patch("models.train_pipeline.build_and_save_all") as build_mock:
            from data.store import save_raw
            from features.build_features import build_and_save_features

            save_raw(ohlcv, "AAPL")
            build_mock.side_effect = lambda symbols: {
                symbol.strip().upper(): build_and_save_features(symbol)
                for symbol in symbols
            }

            result = run_pipeline(["AAPL"], years=2, train_end="2021-06-30")

    assert result["n_rows"] >= 100
    assert (pipeline_dirs / "model_xgb.joblib").exists()
    assert (pipeline_dirs / "model_meta.json").exists()
