"""Tests for ranking pipeline: features, composite, storage, incremental OHLCV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.store import append_raw, merge_ohlcv, raw_exists
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.features.ranking_features import compute_ranking_features
from ranking_pipeline.scoring.composite import score_universe_slice
from ranking_pipeline.storage.sqlite import RankingStore
from ranking_pipeline.universe.filters import compute_adv_dollars, screen_symbol_ohlcv


def _synthetic_ohlcv(rows: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = 100 + np.cumsum(rng.normal(0, 1, size=rows))
    high = close + rng.uniform(0.5, 2, size=rows)
    low = close - rng.uniform(0.5, 2, size=rows)
    open_ = close + rng.normal(0, 0.5, size=rows)
    volume = rng.integers(1_000_000, 5_000_000, size=rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _spy_close(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(100.0, index=index)


def test_compute_ranking_features_columns():
    ohlcv = _synthetic_ohlcv()
    feats = compute_ranking_features(ohlcv, _spy_close(ohlcv.index), include_labels=True)
    assert not feats.empty
    for col in (
        "close_vs_sma20",
        "excess_ret_5d_vs_spy",
        "rel_volume",
        "dist_20d_high",
        "atr_14",
        "pattern_signal_score",
    ):
        assert col in feats.columns


def test_composite_weights_contributions_sum():
    panel = pd.DataFrame(
        {
            "close_vs_sma20": [0.1, -0.2, 0.05],
            "close_vs_sma50": [0.2, 0.0, 0.1],
            "close_vs_sma200": [0.0, 0.1, -0.1],
            "sma20_slope_5d": [0.01, 0.02, -0.01],
            "sma50_slope_5d": [0.02, -0.01, 0.0],
            "excess_ret_5d_vs_spy": [0.03, 0.04, -0.02],
            "excess_ret_20d_vs_spy": [0.05, -0.03, 0.01],
            "excess_ret_60d_vs_spy": [0.08, 0.02, 0.0],
            "rel_volume": [1.2, 0.8, 1.5],
            "vol_ratio_20d": [1.1, 0.9, 1.4],
            "dist_20d_high": [-0.02, -0.05, -0.01],
            "dist_52w_high": [-0.1, -0.2, -0.05],
            "new_high_20d": [0.0, 1.0, 0.0],
            "new_high_52w": [0.0, 0.0, 1.0],
            "atr_14": [2.0, 2.5, 1.8],
            "atr_percentile_252d": [0.5, 0.6, 0.4],
            "pat_engulfing": [0.0, 100.0, 0.0],
            "pat_hammer": [0.0, 0.0, 100.0],
            "pat_morningstar": [0.0, 0.0, 0.0],
            "pattern_signal_score": [0.0, 1.0, 1.0],
        },
        index=["AAA", "BBB", "CCC"],
    )
    cfg = RankingPipelineConfig()
    results = score_universe_slice(panel, cfg)
    weights = cfg.normalized_weights()
    for res in results:
        assert abs(sum(res.contributions.values()) - res.composite_score) < 1e-6
        for group in weights:
            if group in res.contributions:
                assert group in res.contributions


def test_liquidity_screen():
    ohlcv = _synthetic_ohlcv(30)
    ohlcv["close"] = 10.0
    ohlcv["volume"] = 5_000_000
    adv = compute_adv_dollars(ohlcv)
    assert adv == 10.0 * 5_000_000
    metrics = screen_symbol_ohlcv(
        "TEST",
        ohlcv,
        market_cap=2e9,
        filters=RankingPipelineConfig().liquidity,
    )
    assert metrics.passed is True


def test_sqlite_ranking_roundtrip(tmp_path: Path):
    db = tmp_path / "rank.db"
    store = RankingStore(db)
    store.save_universe_snapshot(
        "2026-01-01",
        [
            {
                "symbol": "AAPL",
                "last_close": 200.0,
                "market_cap": 3e12,
                "avg_dollar_volume_20d": 1e9,
                "passed_filters": True,
            }
        ],
    )
    assert store.load_universe_symbols("2026-01-01") == ["AAPL"]
    store.save_ranking_run(
        "run-1",
        "2026-01-02",
        "composite",
        "2026-01-01",
        [
            {
                "symbol": "AAPL",
                "rank": 1,
                "composite_score": 1.5,
                "ml_probability": 0.7,
                "expected_excess_return": 0.02,
                "final_score": 1.2,
                "contributions": {"trend": 0.5, "relative_strength": 0.7},
            }
        ],
    )
    rows = store.get_ranking_results("run-1", limit=5)
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["contributions"]["trend"] == 0.5


def test_merge_ohlcv_dedupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dates = pd.bdate_range("2024-01-01", periods=3)
    first = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
        },
        index=dates,
    )
    second = first.copy()
    second.iloc[-1, second.columns.get_loc("close")] = 99.0
    merged = merge_ohlcv(first, second)
    assert merged.iloc[-1]["close"] == 99.0
    assert len(merged) == 3


def test_incremental_raw_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from data import paths

    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    tmp_path.joinpath("raw").mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        },
        index=dates,
    )
    append_raw(df, "ZZZ")
    assert raw_exists("ZZZ")
    extra = df.iloc[[-1]].copy()
    extra.iloc[0, extra.columns.get_loc("close")] = 5.0
    append_raw(extra, "ZZZ")
    from data.store import load_raw

    loaded = load_raw("ZZZ")
    assert loaded.iloc[-1]["close"] == 5.0
