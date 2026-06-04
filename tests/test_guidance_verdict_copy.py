"""Verdict copy must align with verdict severity."""

from __future__ import annotations

from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.guidance_verdict_copy import build_equity_verdict_copy
from app.models.intelligence_models import IntelligenceSignal
from app.models.trade_decision_models import (
    TradeDecision,
    TradeDecisionReasonBreakdown,
    TradeDecisionRegime,
)

_HOLD_PHRASES = ("no major exit pressures", "continue monitoring", "thesis remains intact")
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
        )
    )
    assert result.verdict in {"TRIM", "REVIEW_SELL", "EXIT"}
    blob = " ".join(
        [result.primary_reason, *result.supporting_factors]
    ).lower()
    for phrase in _TRIM_FORBIDDEN:
        assert phrase not in blob


def test_hold_verdict_copy_supports_monitoring():
    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol="TEST",
            as_of_date="2026-06-01",
            trade_decision=_trade(score=85),
            signals=[],
            alert_reasons=[],
            position_weight_pct=8.0,
            open_profit_loss_pct=12.0,
        )
    )
    assert result.verdict == "HOLD"
    assert result.justification
    assert "monitor" in result.primary_reason.lower() or "intact" in result.primary_reason.lower()


def test_primary_reason_leads_with_justification_label():
    primary, supporting, _ = build_equity_verdict_copy(
        verdict="TRIM",
        justification="EXCESSIVE_CONCENTRATION",
        weight_pct=24.0,
        pnl_pct=-5.0,
        regime_env="FAVORABLE",
        trade_score=70,
        regime_id="risk_on_trend",
        critical_signal=None,
    )
    assert primary.startswith("Excessive concentration:")
    assert "reducing exposure" in primary.lower()
    assert any("weakened" in s.lower() or "reducing" in s.lower() for s in supporting)
