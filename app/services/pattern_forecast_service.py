"""Attach 5-day pattern model forecasts to symbol intelligence responses."""

from __future__ import annotations

from typing import Any

from app.models.intelligence_models import PatternTrendForecast
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
    trade_signal = payload.get("trade_signal")
    if trade_signal is None and up_prob is not None:
        min_up_prob = payload.get("min_up_prob")
        if min_up_prob is not None:
            trade_signal = float(up_prob) >= float(min_up_prob)

    return PatternTrendForecast(
        as_of_date=str(payload["date"]),
        horizon_days=LABEL_HORIZON_DAYS,
        label_scheme=scheme.value,
        prediction=int(payload["prediction"]),
        up_prob=float(up_prob) if up_prob is not None else None,
        trade_signal=bool(trade_signal) if trade_signal is not None else None,
        in_training_universe=bool(payload.get("in_training_universe", False)),
        probabilities=dict(payload.get("probabilities") or {}),
        indicators=dict(payload.get("indicators") or {}),
        model_train_end_date=payload.get("model_train_end_date"),
    )
