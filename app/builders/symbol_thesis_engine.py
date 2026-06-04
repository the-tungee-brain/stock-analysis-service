from __future__ import annotations

from dataclasses import dataclass

from app.models.position_guidance_models import SymbolThesis
from app.models.trade_decision_models import TradeDecision, TradeEnvironment


@dataclass(frozen=True)
class SymbolThesisResult:
    thesis: SymbolThesis
    summary: str


def evaluate_symbol_thesis(
    trade: TradeDecision,
    *,
    trend_bias: str | None = None,
) -> SymbolThesisResult:
    env: TradeEnvironment = trade.regime.trade_environment
    score = trade.trade_quality_score
    bias = (trend_bias or "").lower()

    if env == "AVOID" or score < 40:
        return SymbolThesisResult(
            thesis="BEARISH",
            summary=(
                f"Macro/technical setup is defensive (regime {trade.regime.regime_id or 'n/a'}, "
                f"trade quality {score}/100)."
            ),
        )

    if "bear" in bias and score < 55:
        return SymbolThesisResult(
            thesis="BEARISH",
            summary=(
                f"Price trend is weakening relative to the market while trade quality "
                f"is only {score}/100."
            ),
        )

    if env == "FAVORABLE" and score >= 65 and "bear" not in bias:
        return SymbolThesisResult(
            thesis="BULLISH",
            summary=(
                f"Favorable regime with solid trade quality ({score}/100) and supportive trend."
            ),
        )

    return SymbolThesisResult(
        thesis="NEUTRAL",
        summary=(
            f"Mixed signals — regime is {env.lower()} with trade quality {score}/100; "
            "no strong directional edge."
        ),
    )
