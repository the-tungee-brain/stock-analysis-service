"""Shared production config for tradeable pattern model train, backtest, and serve."""

from __future__ import annotations

from typing import Sequence

from backtest.config import BacktestStrategyConfig
from backtest.production_portfolio import ProductionPortfolioConfig
from data.symbols import get_training_universe, get_universe
from features.feature_groups import phase5_model_features, phase5_model_label
from models.labels import LabelScheme
from models.walk_forward import WalkForwardConfig
from models.xgb_model import XGBModelConfig

PRODUCTION_MODEL_KEY = "C"
PRODUCTION_MODEL_LABEL = phase5_model_label("C")
PRODUCTION_TRAINING_UNIVERSE = "top20"
# Legacy alias used by older scripts/tests.
TRADEABLE_UNIVERSE = PRODUCTION_TRAINING_UNIVERSE
PRODUCTION_LABEL_SCHEME = LabelScheme.BINARY_OUTPERFORM_SPY
PRODUCTION_USE_CLASS_WEIGHTS = True
PRODUCTION_MIN_UP_PROB = 0.65
PRODUCTION_TRADE_COST_BPS = 10.0
PRODUCTION_TRAIN_YEARS = 3
PRODUCTION_TEST_YEARS = 1
PRODUCTION_PORTFOLIO_STRATEGY = "ranking"
PRODUCTION_FEATURE_GROUPS: tuple[str, ...] = ("relative_strength", "trend")

PRODUCTION_PORTFOLIO_UNIVERSE = "top20"
PRODUCTION_PORTFOLIO_TOP_N = 10
PRODUCTION_PORTFOLIO_REBALANCE_DAYS = 5
PRODUCTION_PORTFOLIO_HOLD_DAYS = 5
PRODUCTION_MAX_POSITION_WEIGHT = 0.15


def production_model_config() -> XGBModelConfig:
    return XGBModelConfig(
        label_scheme=PRODUCTION_LABEL_SCHEME,
        use_class_weights=PRODUCTION_USE_CLASS_WEIGHTS,
    )


def production_portfolio_config(
    *,
    universe: str = PRODUCTION_PORTFOLIO_UNIVERSE,
    top_n: int = PRODUCTION_PORTFOLIO_TOP_N,
    rebalance_days: int = PRODUCTION_PORTFOLIO_REBALANCE_DAYS,
    hold_days: int = PRODUCTION_PORTFOLIO_HOLD_DAYS,
    max_position_weight: float = PRODUCTION_MAX_POSITION_WEIGHT,
    trade_cost_bps: float = PRODUCTION_TRADE_COST_BPS,
) -> ProductionPortfolioConfig:
    """Default production ranking portfolio (TOP20, top 10, 5d rebalance)."""
    return ProductionPortfolioConfig(
        universe=universe,
        top_n=top_n,
        rebalance_days=rebalance_days,
        hold_days=hold_days,
        max_position_weight=max_position_weight,
        trade_cost_bps=trade_cost_bps,
    )


def production_strategy_config() -> BacktestStrategyConfig:
    return BacktestStrategyConfig(
        min_up_prob=PRODUCTION_MIN_UP_PROB,
        trade_cost_bps=PRODUCTION_TRADE_COST_BPS,
    )


def production_training_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    """Model C feature subset: relative strength + trend (11 columns on full panel)."""
    return phase5_model_features(list(feature_columns), PRODUCTION_MODEL_KEY)


def production_portfolio_metadata() -> dict[str, object]:
    """Portfolio construction settings exposed to API clients."""
    return {
        "strategy_type": PRODUCTION_PORTFOLIO_STRATEGY,
        "portfolio_universe": PRODUCTION_PORTFOLIO_UNIVERSE,
        "top_n": PRODUCTION_PORTFOLIO_TOP_N,
        "rebalance_days": PRODUCTION_PORTFOLIO_REBALANCE_DAYS,
        "hold_days": PRODUCTION_PORTFOLIO_HOLD_DAYS,
        "max_position_weight": PRODUCTION_MAX_POSITION_WEIGHT,
    }


def resolve_production_symbols(
    *,
    extra_symbols: Sequence[str] | None = None,
    universe: str = PRODUCTION_TRAINING_UNIVERSE,
) -> list[str]:
    """Return production training universe symbols, optionally extended with extras."""
    symbols = get_universe(universe)
    if not extra_symbols:
        return symbols

    seen = {symbol.upper() for symbol in symbols}
    merged = list(symbols)
    for raw in extra_symbols:
        symbol = raw.strip().upper()
        if symbol and symbol not in seen:
            merged.append(symbol)
            seen.add(symbol)
    return merged


def resolve_tradeable_symbols(
    *,
    extra_symbols: Sequence[str] | None = None,
    universe: str = PRODUCTION_TRAINING_UNIVERSE,
) -> list[str]:
    """Backward-compatible alias for ``resolve_production_symbols``."""
    return resolve_production_symbols(extra_symbols=extra_symbols, universe=universe)


def production_walk_forward_config(
    *,
    train_years: int = PRODUCTION_TRAIN_YEARS,
    test_years: int = PRODUCTION_TEST_YEARS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> WalkForwardConfig:
    import pandas as pd

    return WalkForwardConfig(
        train_years=train_years,
        test_years=test_years,
        start_date=pd.Timestamp(start_date) if start_date else None,
        end_date=pd.Timestamp(end_date) if end_date else None,
        label_scheme=PRODUCTION_LABEL_SCHEME.value,
        use_class_weights=PRODUCTION_USE_CLASS_WEIGHTS,
        model_config=production_model_config(),
    )


def production_train_metadata_kwargs(*, universe: str = PRODUCTION_TRAINING_UNIVERSE) -> dict:
    """Keyword args persisted in deployment artifact metadata."""
    return {
        "label_scheme": PRODUCTION_LABEL_SCHEME.value,
        "use_class_weights": PRODUCTION_USE_CLASS_WEIGHTS,
        "min_up_prob": PRODUCTION_MIN_UP_PROB,
        "universe": universe,
        "model_key": PRODUCTION_MODEL_KEY,
        "model_label": PRODUCTION_MODEL_LABEL,
        "feature_groups": list(PRODUCTION_FEATURE_GROUPS),
        **production_portfolio_metadata(),
    }


def format_production_universe(*, universe: str = PRODUCTION_TRAINING_UNIVERSE) -> str:
    return ", ".join(get_training_universe(universe))


def format_tradeable_universe() -> str:
    return format_production_universe()
