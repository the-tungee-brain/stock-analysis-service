"""Pipeline configuration: filters, weights, paths, workers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from data.paths import DEFAULT_RANKING_DB_PATH, RANKING_ARTIFACTS_DIR

from ranking_pipeline.execution_costs import ExecutionCostConfig
from ranking_pipeline.ml.labels import ClassificationTarget


class ModelBackend(str, Enum):
    COMPOSITE_ONLY = "composite"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "relative_strength": 0.40,
    "trend": 0.25,
    "volume": 0.20,
    "breakout": 0.10,
    "pattern": 0.05,
}


@dataclass(frozen=True)
class LiquidityFilters:
    min_price: float = 5.0
    min_market_cap: float = 1e9
    min_avg_dollar_volume_20d: float = 20e6
    screening_lookback_days: int = 60


@dataclass
class RankingPipelineConfig:
    db_path: Path = field(default_factory=lambda: DEFAULT_RANKING_DB_PATH)
    artifacts_dir: Path = field(default_factory=lambda: RANKING_ARTIFACTS_DIR)
    liquidity: LiquidityFilters = field(default_factory=LiquidityFilters)
    group_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_GROUP_WEIGHTS)
    )
    model_backend: ModelBackend = ModelBackend.XGBOOST
    ml_blend_weight: float = 0.6
    composite_blend_weight: float = 0.4
    max_workers: int = 4
    feature_warmup_bars: int = 252
    feature_tail_recompute: int = 320
    top_n_api: int = 20
    keep_runs_days: int = 90
    benchmark_symbol: str = "SPY"
    classification_target: ClassificationTarget = ClassificationTarget.OUTPERFORM_SPY
    feature_decay_halflife_days: float = 10.0
    execution_costs: ExecutionCostConfig = field(default_factory=ExecutionCostConfig)
    backtest_top_n: int = 20
    backtest_hold_days: int = 5
    universe_batch_size: int = 250
    universe_max_workers: int = 2
    universe_memory_log_interval: int = 100
    universe_commit_interval: int = 250

    def normalized_weights(self) -> dict[str, float]:
        total = sum(self.group_weights.values())
        if total <= 0:
            raise ValueError("group_weights must sum to a positive value")
        return {k: v / total for k, v in self.group_weights.items()}


def default_config() -> RankingPipelineConfig:
    cfg = RankingPipelineConfig()
    env_weights = os.getenv("RANKING_COMPOSITE_WEIGHTS")
    if env_weights:
        parsed = json.loads(env_weights)
        cfg.group_weights = {**DEFAULT_GROUP_WEIGHTS, **parsed}
    env_db = os.getenv("RANKING_DB_PATH")
    if env_db:
        cfg.db_path = Path(env_db)
    env_backend = os.getenv("RANKING_MODEL_BACKEND")
    if env_backend:
        cfg.model_backend = ModelBackend(env_backend.strip().lower())
    env_workers = os.getenv("RANKING_MAX_WORKERS")
    if env_workers:
        cfg.max_workers = int(env_workers)
    env_universe_workers = os.getenv("RANKING_UNIVERSE_MAX_WORKERS")
    if env_universe_workers:
        cfg.universe_max_workers = int(env_universe_workers)
    env_universe_batch = os.getenv("RANKING_UNIVERSE_BATCH_SIZE")
    if env_universe_batch:
        cfg.universe_batch_size = int(env_universe_batch)
    env_memory_log = os.getenv("RANKING_UNIVERSE_MEMORY_LOG_INTERVAL")
    if env_memory_log:
        cfg.universe_memory_log_interval = int(env_memory_log)
    env_commit_interval = os.getenv("RANKING_UNIVERSE_COMMIT_INTERVAL")
    if env_commit_interval:
        cfg.universe_commit_interval = int(env_commit_interval)
    env_target = os.getenv("RANKING_CLASSIFICATION_TARGET")
    if env_target:
        cfg.classification_target = ClassificationTarget(env_target.strip().lower())
    env_slip = os.getenv("RANKING_SLIPPAGE_BPS")
    if env_slip:
        cfg.execution_costs = ExecutionCostConfig(
            slippage_bps_per_side=float(env_slip),
            round_trip_sides=cfg.execution_costs.round_trip_sides,
            liquidity_penalty_bps=cfg.execution_costs.liquidity_penalty_bps,
            min_adv_dollars=cfg.execution_costs.min_adv_dollars,
        )
    return cfg
