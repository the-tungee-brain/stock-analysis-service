from __future__ import annotations

from copy import deepcopy

from app.builders.trading_bias_engine import (
    RankingContext,
    TradingBiasInputs,
    evaluate_trading_bias,
)
from ranking_pipeline.regime.constants import REGIME_RISK_OFF, REGIME_RISK_ON_TREND
from tests.test_symbol_intelligence_route import (
    _pattern_intelligence_payload,
    _prediction_payload,
)


def _payload(symbol: str = "AAPL") -> dict:
    payload = deepcopy(_pattern_intelligence_payload(symbol))
    payload["chart_intelligence"]["summary"]["outlook"]["label"] = "Bullish"
    payload["chart_intelligence"]["support_zones"] = [
        {
            "price_low": 94.0,
            "price_high": 95.0,
            "label": "Support: $95",
            "zone_type": "support",
            "touches": 3,
            "strength": 0.8,
        }
    ]
    payload["chart_intelligence"]["resistance_zones"] = [
        {
            "price_low": 105.0,
            "price_high": 106.0,
            "label": "Resistance: $105",
            "zone_type": "resistance",
            "touches": 2,
            "strength": 0.7,
        }
    ]
    return payload


def test_bullish_structure_strong_rs_risk_on_returns_bullish_medium_or_high():
    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=_payload(),
            prediction_payload=_prediction_payload(),
            regime_id=REGIME_RISK_ON_TREND,
            ranking=RankingContext(
                ml_probability=0.72,
                expected_excess_return=0.025,
                rank=25,
                universe_count=1000,
            ),
        )
    )

    assert result.bias == "Bullish"
    assert result.confidence in {"High", "Medium"}
    assert result.horizon == "1-5 sessions"
    assert result.alignment.pattern_trend == "aligned"
    assert result.alignment.relative_strength == "aligned"
    assert result.levels.support == 95.0
    assert result.levels.breakout_level == 105.0


def test_strong_model_c_but_bearish_structure_is_not_bullish():
    payload = _payload()
    payload["trend_context"]["trend_bias"] = "downtrend"
    payload["trend_context"]["above_sma50"] = False
    payload["trend_context"]["above_sma200"] = False
    payload["scores"]["trend_strength"] = 0.2
    payload["chart_intelligence"]["summary"]["outlook"]["label"] = "Bearish"

    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=payload,
            prediction_payload=_prediction_payload(),
            regime_id=REGIME_RISK_ON_TREND,
            ranking=RankingContext(ml_probability=0.82, expected_excess_return=0.03),
        )
    )

    assert result.bias != "Bullish"
    assert result.alignment.relative_strength == "aligned"
    assert result.alignment.pattern_trend == "against"


def test_risk_off_regime_caps_bullish_confidence():
    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=_payload(),
            prediction_payload=_prediction_payload(),
            regime_id=REGIME_RISK_OFF,
            ranking=RankingContext(ml_probability=0.78, expected_excess_return=0.02),
        )
    )

    assert result.bias == "Bullish"
    assert result.confidence != "High"
    assert result.action == "Risk-off"
    assert result.alignment.market_regime == "against"


def test_failed_breakout_distribution_volume_returns_bearish_or_neutral():
    payload = _payload()
    payload["trend_context"]["trend_bias"] = "mixed"
    payload["trend_context"]["rs_vs_spy_63d"] = -0.04
    payload["trend_context"]["vol_ratio_20d"] = 1.6
    payload["scores"]["trend_strength"] = 0.35
    payload["scores"]["volume_confirmation"] = 0.25
    payload["chart_intelligence"]["summary"]["outlook"]["label"] = "Bearish"
    payload["chart_intelligence"]["summary"]["why_this_outlook"] = [
        {"text": "Distribution pressure is visible.", "tone": "caution"}
    ]
    payload["chart_intelligence"]["breakout_events"] = [
        {
            "kind": "failed_breakout",
            "date": "2026-06-04",
            "price": 99.0,
            "zone_label": "Resistance: $105",
            "label": "Failed breakout",
            "volume_ratio": 1.6,
        }
    ]

    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=payload,
            regime_id="risk_on_chop",
            ranking=RankingContext(ml_probability=0.38, expected_excess_return=-0.015),
        )
    )

    assert result.bias in {"Bearish", "Neutral"}
    assert result.alignment.volume == "warning"
    assert "Failed breakout warns of overhead supply" in result.bearish_factors


def test_missing_model_c_still_returns_valid_bias():
    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=_payload(),
            prediction_payload=None,
            regime_id=REGIME_RISK_ON_TREND,
            ranking=None,
            data_gaps=["Model C ranking unavailable"],
        )
    )

    assert result.bias in {"Bullish", "Neutral", "Bearish"}
    assert result.confidence in {"High", "Medium", "Low"}
    assert "Model C ranking unavailable" in result.data_gaps


def test_missing_pattern_analysis_returns_neutral_low_with_data_gap():
    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=None,
            regime_id=REGIME_RISK_ON_TREND,
            data_gaps=["Pattern analysis unavailable"],
        )
    )

    assert result.bias == "Neutral"
    assert result.confidence == "Low"
    assert result.action == "Watch"
    assert result.data_gaps == ["Pattern analysis unavailable"]


def test_response_shape_uses_stable_aliases():
    result = evaluate_trading_bias(
        TradingBiasInputs(
            symbol="AAPL",
            pattern_intelligence=_payload(),
            prediction_payload=_prediction_payload(),
            regime_id=REGIME_RISK_ON_TREND,
        )
    )

    payload = result.model_dump(mode="json", by_alias=True)

    assert set(payload) == {
        "symbol",
        "bias",
        "confidence",
        "horizon",
        "action",
        "bullishFactors",
        "bearishFactors",
        "invalidation",
        "levels",
        "alignment",
        "dataGaps",
    }
    assert set(payload["levels"]) == {
        "support",
        "resistance",
        "breakoutLevel",
        "stopInvalidLevel",
    }
    assert set(payload["alignment"]) == {
        "marketRegime",
        "relativeStrength",
        "patternTrend",
        "volume",
        "catalyst",
    }
