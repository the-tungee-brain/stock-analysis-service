"""Tests for Phase 5 minimal model feature resolution."""

from __future__ import annotations

from features.feature_groups import phase5_model_features, phase5_model_label


def _columns() -> list[str]:
    return [
        "ret_21d",
        "rs_vs_spy_21d",
        "close_vs_sma20",
        "sma_200",
        "spy_ret_5d",
        "vix_level",
        "rsi_14",
        "vol_zscore_20d",
    ]


def test_phase5_model_feature_counts():
    cols = _columns()
    assert len(phase5_model_features(cols, "A")) == 1
    assert len(phase5_model_features(cols, "B")) == 2
    assert len(phase5_model_features(cols, "C")) == 3
    assert len(phase5_model_features(cols, "D")) == 5
    assert "ret_21d" in phase5_model_features(cols, "E")
    assert "close_vs_sma20" not in phase5_model_features(cols, "E")
    assert len(phase5_model_features(cols, "F")) == len(cols)


def test_phase5_model_labels():
    assert "Relative strength only" in phase5_model_label("A")
    assert "Full model" in phase5_model_label("F")
