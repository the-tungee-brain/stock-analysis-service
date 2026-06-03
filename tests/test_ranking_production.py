"""Production realism: regime, costs, backtest, leakage, ML targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.labels import EXCESS_RETURN_COLUMN, FUTURE_RETURN_COLUMN
from ranking_pipeline.backtest.costs import ExecutionCostConfig, net_excess_return
from ranking_pipeline.backtest.metrics import compute_metrics
from ranking_pipeline.backtest.simulator import TradeResult, simulate_top_n_long
from ranking_pipeline.ml.labels import (
    TOP_QUINTILE_LABEL_COLUMN,
    ClassificationTarget,
    add_top_quintile_labels,
    forward_returns_aligned,
)
from ranking_pipeline.regime.constants import REGIME_RISK_OFF
from ranking_pipeline.regime.detector import compute_spy_regime_series, regime_for_date
from ranking_pipeline.regime.multipliers import regime_score_multiplier
from ranking_pipeline.validation.leakage import assert_no_feature_leakage, validate_feature_frame


def _ohlcv(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=rows))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1_000_000, 3_000_000, size=rows),
        },
        index=dates,
    )


def test_regime_multiplier_dampens_risk_off():
    assert regime_score_multiplier(REGIME_RISK_OFF) < regime_score_multiplier("risk_on_trend")


def test_spy_regime_series_has_regime_id():
    df = compute_spy_regime_series(_ohlcv())
    assert "regime_id" in df.columns
    snap = regime_for_date(df, df.index[-1])
    assert snap is not None
    assert snap.regime_multiplier > 0


def test_slippage_reduces_net_excess():
    gross = 0.02
    net = net_excess_return(gross, config=ExecutionCostConfig(slippage_bps_per_side=20))
    assert net < gross


def test_liquidity_penalty_for_low_adv():
    cfg = ExecutionCostConfig(liquidity_penalty_bps=20, min_adv_dollars=20e6)
    high = net_excess_return(0.01, avg_dollar_volume_20d=50e6, config=cfg)
    low = net_excess_return(0.01, avg_dollar_volume_20d=5e6, config=cfg)
    assert low < high


def test_top_quintile_labels_cross_section_only():
    idx = pd.MultiIndex.from_product(
        [["A", "B", "C", "D", "E"], pd.bdate_range("2024-01-01", periods=3)],
        names=["symbol", "date"],
    )
    excess = pd.Series(np.linspace(-0.05, 0.05, len(idx)), index=idx)
    panel = pd.DataFrame({EXCESS_RETURN_COLUMN: excess})
    labeled = add_top_quintile_labels(panel)
    for date in panel.index.get_level_values("date").unique():
        day = labeled.xs(date, level="date")[TOP_QUINTILE_LABEL_COLUMN]
        assert day.sum() >= 1


def test_forward_returns_use_shift_not_lookahead():
    dates = pd.bdate_range("2024-01-01", periods=20)
    close = pd.Series(np.arange(100, 120, dtype="float64"), index=dates)
    spy = close * 0.99
    future_ret, excess, _ = forward_returns_aligned(close, spy, horizon=5)
    assert future_ret.iloc[-5:].isna().all()
    assert pd.notna(future_ret.iloc[0])


def test_backtest_metrics_from_trades():
    trades = [
        TradeResult("A", "2024-06-01", 0.01, 0.005, 0.003, 50e6),
        TradeResult("B", "2024-06-01", 0.02, 0.01, 0.008, 50e6),
    ]
    m = compute_metrics(trades)
    assert m is not None
    assert 0 <= m.hit_rate_vs_spy <= 1


def test_leakage_validation_rejects_recent_labels():
    ohlcv = _ohlcv(50)
    dates = ohlcv.index
    features = pd.DataFrame(
        {FUTURE_RETURN_COLUMN: 0.01, EXCESS_RETURN_COLUMN: 0.005},
        index=dates,
    )
    errors = validate_feature_frame(features, ohlcv)
    assert any("forward-return leak" in e for e in errors)


def test_shifted_high_prevents_same_bar_breakout_leak():
    from ranking_pipeline.features.ranking_features import compute_ranking_features

    dates = pd.bdate_range("2023-01-01", periods=280)
    n = len(dates)
    close = np.linspace(10.0, 12.0, n)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "low": close - 0.5,
            "high": close + 0.5,
            "close": close,
            "volume": np.full(n, 1_000_000),
        },
        index=dates,
    )
    ohlcv.loc[dates[-1], "high"] = 50.0  # intraday spike must not count as 20d high
    spy = pd.Series(10.0, index=dates)
    feats = compute_ranking_features(
        ohlcv,
        spy,
        include_labels=False,
        validate_leakage=False,
    )
    assert not feats.empty
    last = feats.iloc[-1]
    assert float(last["new_high_20d"]) == 0.0
