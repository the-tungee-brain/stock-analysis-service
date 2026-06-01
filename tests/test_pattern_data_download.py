"""Tests for daily OHLCV download and Parquet storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.download import download_and_store_symbol, download_symbol
from data.loader import load_symbol
from data.paths import raw_parquet_path
from data.store import OHLCV_COLUMNS


@pytest.mark.integration
def test_download_symbol_returns_normalized_ohlcv(tmp_path, monkeypatch):
    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yfinance-cache"))

    df = download_symbol("AAPL", years=2)

    assert not df.empty
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing
    assert (df[["open", "high", "low", "close", "volume"]] > 0).all().all()


@pytest.mark.integration
def test_download_and_store_symbol_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yfinance-cache"))
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)

    path, df = download_and_store_symbol("AAPL", years=2)

    assert path == raw_dir / "AAPL.parquet"
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(OHLCV_COLUMNS)


def test_load_symbol_reads_parquet(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    raw_dir.mkdir(parents=True)

    index = pd.date_range("2024-01-01", periods=5, freq="B", name="date")
    sample = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1_000_000] * 5,
        },
        index=index,
    )
    sample.to_parquet(raw_parquet_path("AAPL"))

    loaded = load_symbol("AAPL")
    assert len(loaded) == 5
    assert list(loaded.columns) == list(OHLCV_COLUMNS)
