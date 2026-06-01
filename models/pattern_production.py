"""Shared production config for tradeable pattern model train, backtest, and serve."""

from __future__ import annotations

from typing import Sequence

from backtest.config import BacktestStrategyConfig
from data.symbols import UNIVERSE_TRADEABLE_V1, get_universe
from models.labels import LabelScheme
from models.walk_forward import WalkForwardConfig
from models.xgb_model import XGBModelConfig

TRADEABLE_UNIVERSE = "tradeable_v1"
PRODUCTION_LABEL_SCHEME = LabelScheme.BINARY_UPDOWN
PRODUCTION_USE_CLASS_WEIGHTS = True
PRODUCTION_MIN_UP_PROB = 0.65
PRODUCTION_TRADE_COST_BPS = 10.0
PRODUCTION_TRAIN_YEARS = 3
PRODUCTION_TEST_YEARS = 1


def production_model_config() -> XGBModelConfig:
    return XGBModelConfig(
        label_scheme=PRODUCTION_LABEL_SCHEME,
        use_class_weights=PRODUCTION_USE_CLASS_WEIGHTS,
    )


def production_strategy_config() -> BacktestStrategyConfig:
    return BacktestStrategyConfig(
        min_up_prob=PRODUCTION_MIN_UP_PROB,
        trade_cost_bps=PRODUCTION_TRADE_COST_BPS,
    )


def resolve_tradeable_symbols(
    *,
    extra_symbols: Sequence[str] | None = None,
    universe: str = TRADEABLE_UNIVERSE,
) -> list[str]:
    """Return tradeable universe symbols, optionally extended with extras."""
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


def production_train_metadata_kwargs(*, universe: str = TRADEABLE_UNIVERSE) -> dict:
    """Keyword args for ``build_model_metadata`` on production artifacts."""
    return {
        "label_scheme": PRODUCTION_LABEL_SCHEME.value,
        "use_class_weights": PRODUCTION_USE_CLASS_WEIGHTS,
        "min_up_prob": PRODUCTION_MIN_UP_PROB,
        "universe": universe,
    }


def format_tradeable_universe() -> str:
    return ", ".join(UNIVERSE_TRADEABLE_V1)
