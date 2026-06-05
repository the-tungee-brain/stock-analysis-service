from __future__ import annotations

import logging
from typing import Any

from app.builders.trade_decision_engine import (
    TradeDecisionInputs,
    compute_breakout_quality_score,
    evaluate_trade_decision,
    inputs_from_chart_payload,
    pattern_reliability_from_setup,
)
from app.models.trade_decision_models import TradeDecision
from app.services.pattern_intelligence_service import (
    build_pattern_intelligence_payload,
    pattern_intelligence_from_dict,
)
from models.prediction_service import LoadedModel
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store

logger = logging.getLogger(__name__)


def build_trade_decision(
    symbol: str,
    *,
    loaded_model: LoadedModel | None = None,
    pattern_analysis_service=None,
) -> TradeDecision:
    symbol_upper = symbol.strip().upper()
    cfg = default_config()
    store = open_store(cfg)

    regime_id: str | None = None
    as_of_date: str | None = None
    ranking_rank: int | None = None
    universe_count: int | None = None

    run_id = store.latest_run_id()
    if run_id:
        meta = store.get_run_meta(run_id)
        if meta:
            regime_id = meta.get("regime_id")
            as_of_date = meta.get("as_of_date")
        row = store.get_symbol_ranking_row(run_id, symbol_upper)
        if row:
            ranking_rank = int(row["rank"])
        universe_count = store.count_ranking_results(run_id) or None

    if loaded_model is None:
        try:
            from models.prediction_service import load_deployed_model

            loaded_model = load_deployed_model()
        except Exception:
            loaded_model = None

    pattern, prediction_payload = _load_pattern_analysis(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
    )
    return _decision_from_pattern(
        symbol_upper=symbol_upper,
        as_of_date=as_of_date or (pattern.as_of_date if pattern else None),
        regime_id=regime_id,
        ranking_rank=ranking_rank,
        universe_count=universe_count,
        pattern=pattern,
        prediction_payload=prediction_payload,
    )


def _load_pattern_analysis(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
) -> tuple[Any | None, dict[str, Any] | None]:
    if loaded_model is None:
        return None, None

    if pattern_analysis_service is not None:
        try:
            snapshot = pattern_analysis_service.get_or_build(symbol_upper, loaded_model)
            return (
                pattern_intelligence_from_dict(snapshot.pattern_intelligence),
                dict(snapshot.prediction_payload),
            )
        except (FileNotFoundError, ValueError, OSError):
            return None, None
        except Exception:
            logger.warning(
                "Pattern analysis service unavailable for trade decision: %s",
                symbol_upper,
                exc_info=True,
            )

    pattern = build_pattern_intelligence_payload(symbol_upper, loaded_model)
    prediction_payload = _forecast_indicators_payload(symbol_upper, loaded_model)
    return pattern, prediction_payload


def _decision_from_pattern(
    *,
    symbol_upper: str,
    as_of_date: str | None,
    regime_id: str | None,
    ranking_rank: int | None,
    universe_count: int | None,
    pattern: Any,
    prediction_payload: dict[str, Any] | None = None,
) -> TradeDecision:
    rs_score = None
    rs_21 = None
    rs_63 = None
    vol_ratio = None
    breakout_score = 35
    sr_conf = 45
    reliability = pattern_reliability_from_setup(None, None)
    acceleration = False
    dist_52w = None
    near_high = False

    chart_dict: dict[str, Any] | None = None
    if pattern is not None:
        tc = pattern.trend_context
        rs_21 = tc.rs_vs_spy_21d
        rs_63 = tc.rs_vs_spy_63d
        vol_ratio = tc.vol_ratio_20d
        rs_score = pattern.scores.relative_strength

        if pattern.setup_outcome is not None:
            reliability = pattern_reliability_from_setup(
                pattern.setup_outcome.occurrence_count,
                pattern.setup_outcome.win_rate_5d,
            )
        elif pattern.historical_stats is not None:
            reliability = pattern_reliability_from_setup(
                pattern.historical_stats.occurrence_count,
                pattern.historical_stats.win_rate_5d,
            )

        chart_dict = (
            pattern.chart_intelligence.model_dump(by_alias=True)
            if hasattr(pattern.chart_intelligence, "model_dump")
            else dict(pattern.chart_intelligence or {})
        )
        breakout_score, sr_conf, confirmed, failed = inputs_from_chart_payload(
            chart_dict
        )
        breakout_score = compute_breakout_quality_score(
            volume_confirmed=vol_ratio is not None and vol_ratio >= 1.2,
            confirmed_breakout=confirmed,
            failed_breakout=failed,
            volume_ratio=vol_ratio,
            volume_confirmation_score=pattern.scores.volume_confirmation,
        )

        summary = chart_dict.get("summary") or {}
        for bullet in summary.get("whyThisOutlook") or summary.get("why_this_outlook") or []:
            text = (bullet.get("text") or "").lower()
            if "acceleration" in text:
                acceleration = True

    try:
        from analysis.pattern_intelligence.chart_analysis import analyze_trend_structure
        from models.prediction_service import ensure_raw_ohlcv

        structure = analyze_trend_structure(ensure_raw_ohlcv(symbol_upper))
        acceleration = structure.acceleration
    except Exception:
        pass

    indicators = _forecast_indicators(symbol_upper, prediction_payload=prediction_payload)
    if indicators:
        dist_52w = _float_or_none(
            indicators.get("dist_52w_high") or indicators.get("dist_52w_high_pct")
        )
        if dist_52w is not None and dist_52w <= 0.03:
            near_high = True
        if _float_or_none(indicators.get("new_high_52w")) == 1.0:
            near_high = True

    inputs = TradeDecisionInputs(
        symbol=symbol_upper,
        as_of_date=as_of_date,
        regime_id=regime_id,
        market_breadth_pct=None,
        rs_percentile=None,
        rs_score_0_1=rs_score,
        rs_21d=rs_21,
        rs_63d=rs_63,
        vol_ratio_20d=vol_ratio,
        dist_52w_high_pct=dist_52w,
        near_52w_high=near_high,
        trend_acceleration=acceleration,
        breakout_quality_score=breakout_score,
        support_resistance_confidence=sr_conf,
        pattern_reliability=reliability,
        ranking_rank=ranking_rank,
        universe_rank_count=universe_count,
    )
    return evaluate_trade_decision(inputs)


def _forecast_indicators(
    symbol: str,
    *,
    prediction_payload: dict[str, Any] | None = None,
) -> dict[str, float]:
    if prediction_payload is not None:
        raw = prediction_payload.get("indicators") or {}
        return {k: float(v) for k, v in raw.items() if v is not None}
    try:
        from models.prediction_service import load_deployed_model

        loaded = load_deployed_model()
    except Exception:
        return {}
    return _forecast_indicators_payload(symbol, loaded)


def _forecast_indicators_payload(
    symbol: str,
    loaded: LoadedModel | None,
) -> dict[str, float]:
    try:
        from models.prediction_service import predict_for_symbol

        if loaded is None:
            return {}
        payload = predict_for_symbol(symbol, loaded)
        raw = payload.get("indicators") or {}
        return {k: float(v) for k, v in raw.items() if v is not None}
    except Exception:
        return {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
