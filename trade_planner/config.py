"""Configurable strategy parameters — no magic numbers in setup logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MomentumBreakoutConfig:
    sma_fast_days: int = 50
    sma_slow_days: int = 200
    high_lookback_days: int = 20
    high_proximity_pct: float = 0.02
    volume_avg_days: int = 20
    volume_ratio_min: float = 1.5
    rs_lookback_days: int = 60
    rs_percentile_window: int = 120
    rs_percentile_min: float = 80.0
    stop_lookback_days: int = 10
    entry_buffer: float = 0.01
    target_risk_reward: float = 2.0
    require_benchmark: bool = True


@dataclass(frozen=True, slots=True)
class PullbackConfig:
    trend_ma_days: int = 50
    pullback_ma_days: int = 20
    max_pullback_pct: float = 0.05
    min_pullback_pct: float = 0.01
    atr_period: int = 14
    stop_atr_multiple: float = 1.25
    target_risk_reward: float = 2.0


@dataclass(frozen=True, slots=True)
class TrendContinuationConfig:
    fast_ma_days: int = 20
    slow_ma_days: int = 50
    volume_avg_days: int = 20
    volume_expansion_ratio: float = 1.2
    atr_period: int = 14
    stop_atr_multiple: float = 1.5
    target_risk_reward: float = 2.5
    min_ma_spread_pct: float = 0.01


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    max_holding_days: int = 20
    min_warmup_bars: int = 60
    commission_bps: float = 0.0
    slippage_bps: float = 5.0


@dataclass(frozen=True, slots=True)
class RankingConfig:
    trend_weight: float = 0.35
    relative_strength_weight: float = 0.25
    volume_weight: float = 0.20
    setup_quality_weight: float = 0.20
    trend_lookback_days: int = 60
    rs_lookback_days: int = 60
    volume_avg_days: int = 20


@dataclass(frozen=True, slots=True)
class TradePlannerConfig:
    momentum: MomentumBreakoutConfig = MomentumBreakoutConfig()
    pullback: PullbackConfig = PullbackConfig()
    trend_continuation: TrendContinuationConfig = TrendContinuationConfig()
    backtest: BacktestConfig = BacktestConfig()
    ranking: RankingConfig = RankingConfig()
