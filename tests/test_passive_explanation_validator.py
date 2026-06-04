"""Passive explanation layer must not act as a decision engine."""

from __future__ import annotations

import pytest

from app.builders.passive_explanation_validator import (
    PassiveExplanationViolation,
    validate_passive_explanation_text,
    validate_trace_matches_guidance,
    validate_verdict_in_trace,
)
from app.builders.position_guidance_scoring_trace import build_symbol_scoring_trace
from app.models.position_guidance_models import (
    GuidanceDriverModel,
    PositionGuidanceItem,
    ScoringContributorModel,
    SymbolPositionGuidanceResponse,
)


def test_banned_should_fails():
    with pytest.raises(PassiveExplanationViolation):
        validate_passive_explanation_text("You should close this call.")


def test_banned_i_would_trim_fails():
    with pytest.raises(PassiveExplanationViolation):
        validate_passive_explanation_text("I would trim half the position.")


def test_verdict_line_allowed():
    validate_passive_explanation_text("verdict: HOLD\nurgency: 12/100")


def test_engine_trace_passes_structural_validation():
    item = PositionGuidanceItem(
        positionKey="eq",
        positionKind="EQUITY_LONG",
        displayLabel="AMZN equity",
        instrumentSymbol="AMZN",
        underlyingSymbol="AMZN",
        quantity=1.0,
        marketValue=100.0,
        openProfitLossPct=-2.0,
        verdict="HOLD",
        confidence="medium",
        urgency=18,
        relativeRiskRank=10,
        justification="Stable position",
        primaryDriver=GuidanceDriverModel(
            code="UNFAVORABLE_REGIME",
            label="Unfavorable regime",
            points=12.0,
            detail="Regime risk_on_trend is neutral",
        ),
        primaryReason="Unfavorable regime: Regime risk_on_trend is neutral",
        scoringContributors=[
            ScoringContributorModel(
                bucket="regime",
                points=12.0,
                label="Regime risk_on_trend is neutral",
                driver_code="UNFAVORABLE_REGIME",
            )
        ],
    )
    guidance = SymbolPositionGuidanceResponse(
        symbol="AMZN",
        hasPositions=True,
        positions=[item],
    )
    trace = build_symbol_scoring_trace(guidance)
    validate_trace_matches_guidance(trace, [item])
    validate_verdict_in_trace(
        text=trace, expected_verdict="HOLD", leg_label="AMZN equity"
    )


def test_banned_close_the_call_fails():
    with pytest.raises(PassiveExplanationViolation):
        validate_passive_explanation_text("Close the call before expiry.")


def test_trace_missing_verdict_fails():
    item = PositionGuidanceItem(
        positionKey="eq",
        positionKind="EQUITY_LONG",
        displayLabel="AMZN equity",
        instrumentSymbol="AMZN",
        underlyingSymbol="AMZN",
        quantity=1.0,
        marketValue=100.0,
        verdict="HOLD",
        confidence="medium",
        urgency=10,
        justification="Stable position",
        primaryDriver=GuidanceDriverModel(
            code="STABLE_POSITION",
            label="Stable position",
            points=0.0,
        ),
        primaryReason="Stable position: none",
        scoringContributors=[],
    )
    with pytest.raises(PassiveExplanationViolation):
        validate_trace_matches_guidance("verdict: TRIM only", [item])
