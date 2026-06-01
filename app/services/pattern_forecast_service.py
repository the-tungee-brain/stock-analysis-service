"""Attach 5-day pattern model forecasts to symbol intelligence responses."""

from __future__ import annotations

from typing import Any

from app.models.intelligence_models import PatternPortfolioStrategy, PatternTrendForecast
from models.labels import LABEL_HORIZON_DAYS, LabelScheme, resolve_label_scheme
from models.prediction_service import LoadedModel, predict_for_symbol


def build_pattern_trend_forecast(
    symbol: str,
    loaded: LoadedModel | None,
) -> PatternTrendForecast | None:
    """Return a camelCase-ready forecast payload, or ``None`` when the model is unavailable."""
    if loaded is None:
        return None

    try:
        payload = predict_for_symbol(symbol, loaded)
    except (FileNotFoundError, ValueError):
        return None

    return pattern_forecast_from_prediction(payload)


def pattern_forecast_from_prediction(payload: dict[str, Any]) -> PatternTrendForecast:
    scheme = resolve_label_scheme(
        payload.get("label_scheme", LabelScheme.ORIGINAL_3CLASS.value)
    )
    up_prob = payload.get("up_prob")
    ranking_score = payload.get("ranking_score", up_prob)
    trade_signal = payload.get("trade_signal")
    if trade_signal is None and up_prob is not None:
        min_up_prob = payload.get("min_up_prob")
        if min_up_prob is not None:
            trade_signal = float(up_prob) >= float(min_up_prob)

    portfolio_strategy = None
    raw_strategy = payload.get("portfolio_strategy")
    if isinstance(raw_strategy, dict):
        portfolio_strategy = PatternPortfolioStrategy(
            strategy_type=str(raw_strategy.get("strategy_type", "ranking")),
            universe=str(raw_strategy.get("portfolio_universe", raw_strategy.get("universe", "top20"))),
            top_n=int(raw_strategy.get("top_n", 10)),
            rebalance_days=int(raw_strategy.get("rebalance_days", 5)),
            hold_days=int(raw_strategy.get("hold_days", 5)),
            max_position_weight=float(raw_strategy.get("max_position_weight", 0.15)),
        )

    return PatternTrendForecast(
        as_of_date=str(payload["date"]),
        horizon_days=LABEL_HORIZON_DAYS,
        label_scheme=scheme.value,
        prediction=int(payload["prediction"]),
        up_prob=float(up_prob) if up_prob is not None else None,
        ranking_score=float(ranking_score) if ranking_score is not None else None,
        trade_signal=bool(trade_signal) if trade_signal is not None else None,
        in_training_universe=bool(payload.get("in_training_universe", False)),
        probabilities=dict(payload.get("probabilities") or {}),
        indicators=dict(payload.get("indicators") or {}),
        model_train_end_date=payload.get("model_train_end_date"),
        model_key=payload.get("model_key"),
        model_label=payload.get("model_label"),
        training_universe=payload.get("training_universe"),
        n_features=payload.get("n_features"),
        feature_groups=list(payload.get("feature_groups") or []),
        portfolio_strategy=portfolio_strategy,
    )


def pattern_forecast_to_api_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Serialize a prediction payload with camelCase aliases for HTTP responses."""
    return pattern_forecast_from_prediction(payload).model_dump(mode="json", by_alias=True)
