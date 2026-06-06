from __future__ import annotations

import logging
from typing import Any

from app.builders.trader_playbook_engine import (
    TraderPlaybookInputs,
    evaluate_trader_playbook,
)
from app.models.trade_decision_models import TradeDecision
from app.models.trader_playbook_models import TraderPlaybookResponse
from app.models.trading_bias_models import TradingBiasResponse
from app.services.pattern_intelligence_service import (
    build_pattern_intelligence_payload,
    pattern_intelligence_to_api_dict,
)
from app.services.trade_decision_service import build_trade_decision
from app.services.trading_bias_service import build_trading_bias
from models.prediction_service import LoadedModel

logger = logging.getLogger(__name__)


def build_trader_playbook(
    symbol: str,
    *,
    loaded_model: LoadedModel | None = None,
    pattern_analysis_service=None,
    research_events_service=None,
) -> TraderPlaybookResponse:
    """Build an additive condition-based daily trader playbook."""
    symbol_upper = symbol.strip().upper()
    data_gaps: list[str] = []
    warnings: list[str] = []

    trading_bias = _load_trading_bias(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        research_events_service=research_events_service,
        data_gaps=data_gaps,
    )
    trade_decision = _load_trade_decision(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        data_gaps=data_gaps,
    )
    pattern_payload = _load_pattern_payload(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        data_gaps=data_gaps,
    )
    catalyst = _load_catalyst_alignment(
        symbol_upper,
        research_events_service=research_events_service,
    )

    return evaluate_trader_playbook(
        TraderPlaybookInputs(
            symbol=symbol_upper,
            trading_bias=trading_bias,
            trade_decision=trade_decision,
            pattern_intelligence=pattern_payload,
            catalyst=catalyst,
            data_gaps=data_gaps,
            warnings=warnings,
        )
    )


def _load_trading_bias(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
    research_events_service,
    data_gaps: list[str],
) -> TradingBiasResponse | None:
    try:
        return build_trading_bias(
            symbol_upper,
            loaded_model=loaded_model,
            pattern_analysis_service=pattern_analysis_service,
            research_events_service=research_events_service,
        )
    except Exception:
        logger.warning("Trader playbook trading bias unavailable", exc_info=True)
        data_gaps.append("Trading bias unavailable")
        return None


def _load_trade_decision(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
    data_gaps: list[str],
) -> TradeDecision | None:
    try:
        return build_trade_decision(
            symbol_upper,
            loaded_model=loaded_model,
            pattern_analysis_service=pattern_analysis_service,
        )
    except Exception:
        logger.warning("Trader playbook execution readiness unavailable", exc_info=True)
        data_gaps.append("Execution readiness unavailable")
        return None


def _load_pattern_payload(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
    data_gaps: list[str],
) -> dict[str, Any] | None:
    if loaded_model is not None and pattern_analysis_service is not None:
        try:
            snapshot = pattern_analysis_service.get_or_build(symbol_upper, loaded_model)
            return dict(snapshot.pattern_intelligence)
        except (FileNotFoundError, ValueError, OSError):
            data_gaps.append("Pattern analysis unavailable")
            return None
        except Exception:
            logger.warning(
                "Trader playbook pattern analysis unavailable for %s",
                symbol_upper,
                exc_info=True,
            )
            data_gaps.append("Pattern analysis unavailable")
            return None

    try:
        pattern = build_pattern_intelligence_payload(symbol_upper, loaded_model)
        if pattern is None:
            data_gaps.append("Pattern analysis unavailable")
            return None
        return pattern_intelligence_to_api_dict(pattern)
    except Exception:
        logger.warning(
            "Trader playbook pattern fallback unavailable for %s",
            symbol_upper,
            exc_info=True,
        )
        data_gaps.append("Pattern analysis unavailable")
        return None


def _load_catalyst_alignment(
    symbol_upper: str,
    *,
    research_events_service,
) -> str:
    if research_events_service is None:
        return "none"
    try:
        events = list(research_events_service.get_events(symbol=symbol_upper) or [])[:5]
    except Exception:
        logger.warning(
            "Trader playbook catalyst context unavailable for %s",
            symbol_upper,
            exc_info=True,
        )
        return "none"
    text = " ".join(
        f"{getattr(event, 'title', '')} {getattr(event, 'detail', '')}".lower()
        for event in events
    )
    if not text.strip():
        return "none"
    positive_terms = ("beat", "raise", "raised", "upgrade", "approval", "growth")
    negative_terms = ("miss", "cut", "downgrade", "probe", "lawsuit", "warning")
    positive = sum(1 for term in positive_terms if term in text)
    negative = sum(1 for term in negative_terms if term in text)
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"
