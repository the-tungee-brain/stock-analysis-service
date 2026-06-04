"""Verdict copy must align with verdict severity and scoring contributors."""

from __future__ import annotations

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.guidance_scoring_drivers import build_strict_guidance_copy
from app.builders.guidance_scoring_types import GuidanceDriver, ScoreContributor
from app.models.intelligence_models import IntelligenceSignal
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)

_TRIM_FORBIDDEN = ("no major exit pressures", "continue monitoring only")


def _trade(*, score: int = 72, env: str = "FAVORABLE") -> TradeDecision:
    return TradeDecision(
        symbol="TEST",
        as_of_date="2026-06-01",
        regime=TradeDecisionRegime(regimeId="risk_on_trend", tradeEnvironment=env),
        tradeQualityScore=score,
        scoreBucket="SETUP",
        verdict="WATCHLIST",
        action="WAIT_FOR_SETUP",
        reasonBreakdown=TradeDecisionReasonBreakdown(
            hardBlockers=[],
            primaryWeakness=None,
            secondaryFactors=[],
        ),
    )


def test_trim_verdict_copy_never_sounds_like_hold():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=75),
            signals=[
                IntelligenceSignal(
                    kind="position_size",
                    severity="watch",
                    message="TEST position weight is 22.0% of portfolio.",
                    symbol="TEST",
                )
            ],
            alert_reasons=[],
            position_weight_pct=22.0,
            open_profit_loss_pct=5.0,
            position_quantity=50.0,
        )
    )
    assert result.verdict in {"TRIM", "REVIEW_SELL", "EXIT"}
    blob = " ".join([result.primary_reason, *result.supporting_factors]).lower()
    for phrase in _TRIM_FORBIDDEN:
        assert phrase not in blob


def test_hold_verdict_uses_top_contributor_in_primary_reason():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=85),
            signals=[],
            alert_reasons=[],
            position_weight_pct=8.0,
            open_profit_loss_pct=12.0,
            position_quantity=10.0,
        )
    )
    assert result.verdict == "HOLD"
    top = max(result.contributors, key=lambda c: c.points, default=None)
    if top and top.points > 0:
        assert top.label in result.primary_reason
    else:
        assert result.primary_reason


def test_strict_copy_uses_driver_detail():
    primary, supporting, _ = build_strict_guidance_copy(
        primary=GuidanceDriver(
            code="EXCESSIVE_CONCENTRATION",
            label="Excessive concentration",
            points=15.0,
            detail="Portfolio weight 24.0% is elevated",
        ),
        secondary=None,
        tertiary=None,
        contributors=[
            ScoreContributor(
                bucket="concentration",
                points=15.0,
                label="Portfolio weight 24.0% is elevated",
                driver="EXCESSIVE_CONCENTRATION",
            )
        ],
    )
    assert primary.startswith("Excessive concentration:")
    assert "Portfolio weight 24.0%" in primary
    assert not supporting
