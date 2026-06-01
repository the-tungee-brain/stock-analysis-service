"""Tests for daily feature computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.store import OHLCV_COLUMNS, save_raw
from features.build_features import (
    FEATURE_WARMUP_DAYS,
    build_and_save_features,
    build_features,
    features_ready_slice,
)


def _synthetic_ohlcv(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    index = pd.date_range("2018-01-01", periods=rows, freq="B", name="date")
    close = 100 + np.cumsum(rng.normal(0, 1, size=rows))
    close = np.maximum(close, 1.0)
    spread = rng.uniform(0.5, 2.0, size=rows)
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 1, size=rows),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, size=rows),
        },
        index=index,
    )
    return df.loc[:, list(OHLCV_COLUMNS)]


def test_build_features_has_expected_columns():
    features = build_features(_synthetic_ohlcv())

    assert features.index.name == "date"
    assert "ret_1d" in features.columns
    assert "rsi_14" in features.columns
    assert "sma_200" in features.columns
    assert "macd_hist" in features.columns
    assert "bb_upper" in features.columns
    assert "atr_14" in features.columns
    assert "pat_doji" in features.columns
    assert "ret_21d" in features.columns
    assert "ret_252d" in features.columns
    assert "vol_ratio_20d" in features.columns
    assert "vol_zscore_20d" in features.columns


def test_features_have_no_nans_after_warmup():
    features = build_features(_synthetic_ohlcv())
    ready = features_ready_slice(features)

    assert len(ready) > 0
    assert not ready.isna().any().any()


def test_build_and_save_features_writes_parquet(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)

    save_raw(_synthetic_ohlcv(), "AAPL")
    features = build_and_save_features("AAPL")

    out_path = features_dir / "AAPL.parquet"
    assert out_path.exists()
    assert len(features) > FEATURE_WARMUP_DAYS
    assert not features_ready_slice(features).isna().any().any()


@pytest.mark.integration
def test_build_features_from_downloaded_aapl(tmp_path, monkeypatch):
    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yfinance-cache"))
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr("data.paths.FEATURES_DIR", tmp_path / "features")

    from data.download import download_and_store_symbol

    download_and_store_symbol("AAPL", years=2)
    features = build_and_save_features("AAPL")
    ready = features_ready_slice(features)

    assert len(ready) > 0
    assert not ready.isna().any().any()
