from __future__ import annotations

import logging
from typing import Any

from app.builders.trading_bias_engine import (
    RankingContext,
    TradingBiasInputs,
    evaluate_trading_bias,
)
from app.models.trading_bias_models import TradingBiasResponse
from app.services.pattern_intelligence_service import (
    build_pattern_intelligence_payload,
    pattern_intelligence_to_api_dict,
)
from models.prediction_service import LoadedModel
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def build_trading_bias(
    symbol: str,
    *,
    loaded_model: LoadedModel | None = None,
    pattern_analysis_service=None,
    research_events_service=None,
) -> TradingBiasResponse:
    """Build an additive short-term daily bias from existing research signals."""
    symbol_upper = symbol.strip().upper()
    data_gaps: list[str] = []

    regime_id, ranking = _load_ranking_context(symbol_upper, data_gaps=data_gaps)
    pattern_payload, prediction_payload = _load_pattern_context(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        data_gaps=data_gaps,
    )
    events = _load_recent_events(
        symbol_upper,
        research_events_service=research_events_service,
        data_gaps=data_gaps,
    )

    return evaluate_trading_bias(
        TradingBiasInputs(
            symbol=symbol_upper,
            pattern_intelligence=pattern_payload,
            prediction_payload=prediction_payload,
            regime_id=regime_id,
            ranking=ranking,
            events=events,
            data_gaps=data_gaps,
        )
    )


def _load_ranking_context(
    symbol_upper: str,
    *,
    data_gaps: list[str],
) -> tuple[str | None, RankingContext | None]:
    try:
        store = open_store(default_config())
        run_id = store.latest_run_id()
        if not run_id:
            data_gaps.append("Ranking run unavailable")
            return None, None

        meta = store.get_run_meta(run_id) or {}
        row = store.get_symbol_ranking_row(run_id, symbol_upper)
        universe_count = store.count_ranking_results(run_id) or None
        if not row:
            data_gaps.append("Model C ranking unavailable")
            return meta.get("regime_id"), None

        return (
            meta.get("regime_id"),
            RankingContext(
                ml_probability=_float_or_none(row.get("ml_probability")),
                expected_excess_return=_float_or_none(
                    row.get("expected_excess_return")
                ),
                final_score=_float_or_none(row.get("final_score")),
                rank=_int_or_none(row.get("rank")),
                universe_count=_int_or_none(universe_count),
            ),
        )
    except Exception:
        logger.warning("Trading bias ranking context unavailable", exc_info=True)
        data_gaps.append("Ranking context unavailable")
        return None, None


def _load_pattern_context(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
    data_gaps: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if loaded_model is not None and pattern_analysis_service is not None:
        try:
            snapshot = pattern_analysis_service.get_or_build(symbol_upper, loaded_model)
            return dict(snapshot.pattern_intelligence), dict(snapshot.prediction_payload)
        except (FileNotFoundError, ValueError, OSError):
            data_gaps.append("Pattern analysis unavailable")
        except Exception:
            logger.warning(
                "Pattern analysis service unavailable for trading bias: %s",
                symbol_upper,
                exc_info=True,
            )
            data_gaps.append("Pattern analysis unavailable")

    try:
        pattern = build_pattern_intelligence_payload(symbol_upper, loaded_model)
        if pattern is None:
            data_gaps.append("Pattern analysis unavailable")
            return None, None
        pattern_payload = pattern_intelligence_to_api_dict(pattern)
        return pattern_payload, _pattern_core_model_payload(pattern_payload)
    except Exception:
        logger.warning(
            "Pattern intelligence fallback unavailable for trading bias: %s",
            symbol_upper,
            exc_info=True,
        )
        data_gaps.append("Pattern analysis unavailable")
        return None, None


def _load_recent_events(
    symbol_upper: str,
    *,
    research_events_service,
    data_gaps: list[str],
) -> list[Any]:
    if research_events_service is None:
        return []
    try:
        return list(research_events_service.get_events(symbol=symbol_upper) or [])
    except Exception:
        logger.warning("Trading bias events context unavailable", exc_info=True)
        data_gaps.append("Recent events unavailable")
        return []


def _pattern_core_model_payload(pattern_payload: dict[str, Any]) -> dict[str, Any] | None:
    core = pattern_payload.get("coreModel") or pattern_payload.get("core_model")
    return dict(core) if isinstance(core, dict) else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
