from __future__ import annotations

from dataclasses import dataclass

from app.builders.guidance_verdict_copy import (
    VerdictJustification,
    build_equity_verdict_copy,
    detect_equity_justification,

)
from app.models.position_guidance_models import GuidanceConfidence, EquityVerdict

ExitConfidence = GuidanceConfidence
ExitVerdict = EquityVerdict

_EXIT_DISCLAIMER = (
    "Decision support only — not investment advice or a trade recommendation."
)
from app.models.intelligence_models import IntelligenceSignal
from app.models.trade_decision_models import TradeDecision, TradeEnvironment


@dataclass(frozen=True)
class EquityExitGuidanceInputs:
    symbol: str
    as_of_date: str | None
    trade_decision: TradeDecision
    signals: list[IntelligenceSignal]
    alert_reasons: list[str]
    position_weight_pct: float | None
    open_profit_loss_pct: float | None
    failed_breakout: bool = False
    trend_bias: str | None = None
    rs_vs_spy_21d: float | None = None
    rs_vs_spy_63d: float | None = None
    ranking_rank: int | None = None


@dataclass(frozen=True)
class EquityExitGuidanceResult:
    symbol: str
    as_of_date: str | None
    verdict: ExitVerdict
    confidence: ExitConfidence
    exit_urgency: int
    justification: VerdictJustification
    primary_reason: str
    supporting_factors: list[str]
    risk_factors: list[str]
    would_improve: list[str]
    would_worsen: list[str]
    disclaimer: str = _EXIT_DISCLAIMER


def evaluate_equity_exit_guidance(inputs: EquityExitGuidanceInputs) -> EquityExitGuidanceResult:
    td = inputs.trade_decision
    regime_env = td.regime.trade_environment
    trade_score = td.trade_quality_score

    regime_pts = _regime_stress(regime_env)
    position_pts = _position_stress(
        weight_pct=inputs.position_weight_pct,
        pnl_pct=inputs.open_profit_loss_pct,
    )
    technical_pts = min(25, max(0, 100 - trade_score) * 0.25)
    if inputs.failed_breakout:
        technical_pts = min(25, technical_pts + 10)
    if _is_late_trend(td, inputs):
        technical_pts = min(25, technical_pts + 8)

    rs_pts = _rs_stress(td, inputs)
    volume_pts = _volume_stress(inputs)
    signal_pts = _signal_stress(inputs.signals, inputs.alert_reasons)

    urgency = min(
        100,
        int(round(regime_pts + position_pts + technical_pts + rs_pts + volume_pts + signal_pts)),
    )

    verdict = _verdict_from_urgency(urgency)
    verdict = _apply_floors(
        verdict=verdict,
        urgency=urgency,
        regime_env=regime_env,
        weight_pct=inputs.position_weight_pct,
        pnl_pct=inputs.open_profit_loss_pct,
        signals=inputs.signals,
        alert_reasons=inputs.alert_reasons,
        trade_score=trade_score,
    )
    verdict = _apply_ceilings(
        verdict=verdict,
        trade_score=trade_score,
        weight_pct=inputs.position_weight_pct,
        signals=inputs.signals,
        alert_reasons=inputs.alert_reasons,
    )

    confidence = _confidence(inputs, trade_score, regime_env)

    critical_signal = None
    for signal in inputs.signals:
        if signal.severity == "critical":
            critical_signal = signal.message
            break

    justification = detect_equity_justification(
        signals=inputs.signals,
        alert_reasons=inputs.alert_reasons,
        weight_pct=inputs.position_weight_pct,
        pnl_pct=inputs.open_profit_loss_pct,
        regime_env=regime_env,
        failed_breakout=inputs.failed_breakout,
        trend_bias=inputs.trend_bias,
    )
    primary, supporting, risks = build_equity_verdict_copy(
        verdict=verdict,
        justification=justification,
        weight_pct=inputs.position_weight_pct,
        pnl_pct=inputs.open_profit_loss_pct,
        regime_env=regime_env,
        trade_score=trade_score,
        regime_id=td.regime.regime_id,
        critical_signal=critical_signal,
    )
    improve = _would_improve(inputs, regime_env, trade_score)
    worsen = _would_worsen(inputs, regime_env)

    return EquityExitGuidanceResult(
        symbol=inputs.symbol.upper(),
        as_of_date=inputs.as_of_date,
        verdict=verdict,
        confidence=confidence,
        exit_urgency=urgency,
        justification=justification,
        primary_reason=primary,
        supporting_factors=supporting,
        risk_factors=risks,
        would_improve=improve[:3],
        would_worsen=worsen[:3],
    )


def _regime_stress(env: TradeEnvironment) -> float:
    if env == "AVOID":
        return 25.0
    if env == "NEUTRAL":
        return 12.0
    return 0.0


def _position_stress(*, weight_pct: float | None, pnl_pct: float | None) -> float:
    pts = 0.0
    if weight_pct is not None:
        if weight_pct >= 30:
            pts += 25.0
        elif weight_pct >= 20:
            pts += 15.0
        elif weight_pct >= 15:
            pts += 8.0
    if pnl_pct is not None:
        if pnl_pct <= -30:
            pts = max(pts, 25.0)
        elif pnl_pct <= -20:
            pts = max(pts, 18.0)
        elif pnl_pct <= -10:
            pts = max(pts, 8.0)
    return min(25.0, pts)


def _rs_stress(td: TradeDecision, inputs: EquityExitGuidanceInputs) -> float:
    from app.builders.trade_decision_engine import (
        TradeDecisionInputs,
        _resolve_rs_percentile,
        _trend_stage,
    )

    pseudo = TradeDecisionInputs(
        symbol=inputs.symbol,
        as_of_date=inputs.as_of_date,
        regime_id=td.regime.regime_id,
        market_breadth_pct=None,
        rs_percentile=None,
        rs_score_0_1=None,
        rs_21d=inputs.rs_vs_spy_21d,
        rs_63d=inputs.rs_vs_spy_63d,
        vol_ratio_20d=None,
        dist_52w_high_pct=None,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=50,
        support_resistance_confidence=50,
        pattern_reliability="medium",
        ranking_rank=inputs.ranking_rank,
        universe_rank_count=None,
    )
    rs_pct = _resolve_rs_percentile(pseudo)
    pts = 0.0
    if rs_pct is not None:
        if rs_pct < 40:
            pts = 15.0
        elif rs_pct < 55:
            pts = 10.0
        elif rs_pct < 70:
            pts = 5.0
    bias = (inputs.trend_bias or "").lower()
    if "bear" in bias:
        pts = min(15.0, pts + 5.0)
    if (
        inputs.rs_vs_spy_21d is not None
        and inputs.rs_vs_spy_63d is not None
        and inputs.rs_vs_spy_21d < 0
        and inputs.rs_vs_spy_63d > 0
    ):
        pts = min(15.0, pts + 5.0)
    if _trend_stage(pseudo, rs_pct) == "late":
        pts = min(15.0, pts + 4.0)
    return pts


def _volume_stress(inputs: EquityExitGuidanceInputs) -> float:
    pts = 0.0
    for signal in inputs.signals:
        if signal.kind == "momentum" and signal.severity in {"watch", "warning", "critical"}:
            pts = max(pts, 10.0)
    return min(10.0, pts)


def _signal_stress(
    signals: list[IntelligenceSignal],
    alert_reasons: list[str],
) -> float:
    severity_pts = {"critical": 10.0, "warning": 7.0, "watch": 4.0, "info": 0.0}
    pts = 0.0
    stress_kinds = {
        "drawdown",
        "concentration",
        "position_size",
        "earnings",
        "thesis_drift",
        "valuation",
        "momentum",
    }
    for signal in signals:
        if signal.kind in stress_kinds:
            pts = max(pts, severity_pts.get(signal.severity, 0.0))
    if alert_reasons and pts < 7.0:
        pts = max(pts, 7.0)
    return min(10.0, pts)


def _verdict_from_urgency(urgency: int) -> ExitVerdict:
    if urgency <= 24:
        return "HOLD"
    if urgency <= 44:
        return "TRIM"
    if urgency <= 64:
        return "REVIEW_SELL"
    return "EXIT"


_VERDICT_RANK = {"HOLD": 0, "TRIM": 1, "REVIEW_SELL": 2, "EXIT": 3}


def _max_verdict(a: ExitVerdict, b: ExitVerdict) -> ExitVerdict:
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


def _apply_floors(
    *,
    verdict: ExitVerdict,
    urgency: int,
    regime_env: TradeEnvironment,
    weight_pct: float | None,
    pnl_pct: float | None,
    signals: list[IntelligenceSignal],
    alert_reasons: list[str],
    trade_score: int,
) -> ExitVerdict:
    del urgency, alert_reasons, trade_score
    result = verdict

    if weight_pct is not None and weight_pct >= 30:
        result = _max_verdict(result, "TRIM")

    if pnl_pct is not None and pnl_pct <= -30:
        result = _max_verdict(result, "REVIEW_SELL")

    if pnl_pct is not None and pnl_pct <= -30 and (
        regime_env == "AVOID" or _has_weak_rs(signals)
    ):
        result = _max_verdict(result, "EXIT")

    if regime_env == "AVOID" and pnl_pct is not None and pnl_pct <= -15:
        result = _max_verdict(result, "REVIEW_SELL")

    if regime_env == "AVOID" and pnl_pct is not None and (
        pnl_pct <= -25 or (weight_pct is not None and weight_pct >= 25)
    ):
        result = _max_verdict(result, "EXIT")

    for signal in signals:
        if signal.kind in {"drawdown", "position_size"} and signal.severity == "critical":
            result = _max_verdict(result, "REVIEW_SELL")

    if _earnings_imminent_underwater(signals, pnl_pct):
        result = _max_verdict(result, "TRIM")

    return result


def _apply_ceilings(
    *,
    verdict: ExitVerdict,
    trade_score: int,
    weight_pct: float | None,
    signals: list[IntelligenceSignal],
    alert_reasons: list[str],
) -> ExitVerdict:
    if verdict in {"REVIEW_SELL", "EXIT"}:
        return verdict
    if trade_score < 80:
        return verdict
    if weight_pct is not None and weight_pct >= 20:
        return verdict
    if alert_reasons:
        return verdict
    if any(s.severity in {"critical", "warning"} for s in signals):
        return verdict
    return "HOLD"


def _has_weak_rs(signals: list[IntelligenceSignal]) -> bool:
    return any(s.kind == "momentum" and s.severity in {"warning", "critical"} for s in signals)


def _earnings_imminent_underwater(
    signals: list[IntelligenceSignal],
    pnl_pct: float | None,
) -> bool:
    if pnl_pct is None or pnl_pct > -10:
        return False
    for signal in signals:
        if signal.kind == "earnings" and signal.severity in {"warning", "critical", "watch"}:
            if "1 day" in signal.message.lower() or "2 day" in signal.message.lower():
                return True
            if "3 day" in signal.message.lower():
                return True
    return False


def _confidence(
    inputs: EquityExitGuidanceInputs,
    trade_score: int,
    regime_env: TradeEnvironment,
) -> ExitConfidence:
    gaps = 0
    if inputs.position_weight_pct is None:
        gaps += 1
    if inputs.open_profit_loss_pct is None:
        gaps += 1
    if inputs.trend_bias is None:
        gaps += 1
    if gaps >= 2:
        return "low"

    stress_kinds = sum(
        1
        for s in inputs.signals
        if s.severity in {"warning", "critical"}
        and s.kind
        in {"drawdown", "concentration", "position_size", "earnings", "thesis_drift"}
    )
    if stress_kinds >= 2 and regime_env == "AVOID":
        return "high"
    if trade_score >= 70 and stress_kinds == 0 and regime_env == "FAVORABLE":
        return "high"
    if stress_kinds >= 1 or inputs.alert_reasons:
        return "medium"
    return "medium"


def _would_improve(
    inputs: EquityExitGuidanceInputs,
    regime_env: TradeEnvironment,
    trade_score: int,
) -> list[str]:
    items: list[str] = []
    if regime_env != "FAVORABLE":
        items.append("Regime shifts to risk_on_trend (favorable)")
    if trade_score < 80:
        items.append("Trade quality score recovers above 80")
    if inputs.position_weight_pct is not None and inputs.position_weight_pct >= 15:
        items.append("Portfolio weight falls below 15%")
    if inputs.open_profit_loss_pct is not None and inputs.open_profit_loss_pct < 0:
        items.append("Unrealized loss recovers above -10%")
    items.append("RS percentile holds above 60")
    return items


def _would_worsen(
    inputs: EquityExitGuidanceInputs,
    regime_env: TradeEnvironment,
) -> list[str]:
    items: list[str] = []
    if regime_env != "AVOID":
        items.append("Regime shifts to risk_off or high_vol_chop")
    items.append("Unrealized loss beyond -20%")
    if inputs.failed_breakout:
        items.append("Breakdown below key support on rising volume")
    items.append("Earnings gap down within 3 sessions")
    return items


def _is_late_trend(td: TradeDecision, inputs: EquityExitGuidanceInputs) -> bool:
    from app.builders.trade_decision_engine import TradeDecisionInputs, _trend_stage

    pseudo = TradeDecisionInputs(
        symbol=inputs.symbol,
        as_of_date=inputs.as_of_date,
        regime_id=td.regime.regime_id,
        market_breadth_pct=None,
        rs_percentile=None,
        rs_score_0_1=None,
        rs_21d=inputs.rs_vs_spy_21d,
        rs_63d=inputs.rs_vs_spy_63d,
        vol_ratio_20d=None,
        dist_52w_high_pct=None,
        near_52w_high=False,
        trend_acceleration=False,
        breakout_quality_score=50,
        support_resistance_confidence=50,
        pattern_reliability="medium",
        ranking_rank=inputs.ranking_rank,
        universe_rank_count=None,
    )
    return _trend_stage(pseudo, None) == "late"
