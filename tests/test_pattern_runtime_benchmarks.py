from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, load_benchmark_ohlcv
from data.store import OHLCV_COLUMNS, save_raw
from models.train_pipeline import symbols_with_benchmarks
from scripts import smoke_pattern_runtime


def _ohlcv(rows: int = 30, close_start: float = 100.0, volume: int = 1_000_000) -> pd.DataFrame:
    close = np.linspace(close_start, close_start + 5.0, rows)
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [volume] * rows,
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="B", name="date"),
    ).loc[:, list(OHLCV_COLUMNS)]


def test_vix_benchmark_loader_accepts_local_vix_alias(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_ohlcv(close_start=18.0, volume=0), "VIX")

    loaded = load_benchmark_ohlcv(VIX_SYMBOL)

    assert not loaded.empty
    assert loaded["close"].iloc[0] == pytest.approx(18.0)
    assert loaded["volume"].iloc[0] == 0


def test_pattern_pipeline_download_symbols_include_required_benchmarks():
    assert symbols_with_benchmarks(["AAPL", BENCHMARK_SYMBOL]) == [
        "AAPL",
        BENCHMARK_SYMBOL,
        VIX_SYMBOL,
    ]


def test_smoke_precheck_reports_missing_vix_benchmark(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    monkeypatch.setattr(smoke_pattern_runtime, "ensure_benchmark_ohlcv", lambda: None)
    save_raw(_ohlcv(close_start=500.0), "NVDA")
    save_raw(_ohlcv(close_start=400.0), BENCHMARK_SYMBOL)

    with pytest.raises(RuntimeError, match=r"Required benchmark OHLCV is missing or empty for \^VIX"):
        smoke_pattern_runtime._check_runtime_ohlcv("NVDA")
