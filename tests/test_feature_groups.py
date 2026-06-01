"""Tests for feature group classification."""

from __future__ import annotations

from features.feature_groups import (
    classify_feature,
    group_features,
    simplicity_feature_columns,
    without_group,
)


def test_classify_core_feature_families():
    assert classify_feature("rs_vs_spy_21d") == "relative_strength"
    assert classify_feature("ret_63d") == "momentum"
    assert classify_feature("close_vs_sma200") == "trend"
    assert classify_feature("vol_zscore_20d") == "volume"
    assert classify_feature("spy_ret_5d") == "market_context"
    assert classify_feature("vix_level") == "market_context"
    assert classify_feature("rsi_14") == "technical"


def test_simplicity_features_exclude_technical_and_volume():
    columns = [
        "ret_21d",
        "rs_vs_spy_21d",
        "spy_ret_5d",
        "vix_level",
        "rsi_14",
        "vol_zscore_20d",
        "close_vs_sma20",
    ]
    simple = simplicity_feature_columns(columns)
    assert "rsi_14" not in simple
    assert "vol_zscore_20d" not in simple
    assert "close_vs_sma20" not in simple
    assert "rs_vs_spy_21d" in simple
    assert "ret_21d" in simple
    assert "spy_ret_5d" in simple


def test_without_group_removes_one_family():
    columns = ["ret_21d", "rs_vs_spy_21d", "vol_zscore_20d"]
    remaining = without_group(columns, "volume")
    assert remaining == ["ret_21d", "rs_vs_spy_21d"]
