from __future__ import annotations

import logging
from typing import Any

from app.broker.position_metrics import position_open_profit_loss_pct
from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.trade_decision_engine import inputs_from_chart_payload
from app.models.equity_exit_guidance_models import (
    EquityExitGuidance,
    EquityExitGuidanceContext,
    PortfolioExitAttentionItem,
    PortfolioExitAttentionResponse,
)
from app.models.intelligence_models import IntelligenceSignal, ProactiveAlert
from app.models.schwab_models import Position, SchwabAccounts
from app.services.company_research_service import CompanyResearchService
from app.services.intelligence.signal_engine import SignalEngine, build_proactive_alerts
from app.services.portfolio_service import PortfolioService
from app.services.trade_decision_service import build_trade_decision

logger = logging.getLogger(__name__)

_ATTENTION_VERDICTS = frozenset({"TRIM", "REVIEW_SELL", "EXIT"})


def _is_equity_long(position: Position, symbol_upper: str) -> bool:
    if position.longQuantity <= 0:
        return False
    instrument = position.instrument
    asset = (instrument.assetType or "").upper()
    if asset == "OPTION" or instrument.putCall:
        return False
    sym = (instrument.symbol or "").upper()
    if instrument.underlyingSymbol:
        return False
    return sym == symbol_upper


def _scoped_equity_positions(
    positions: list[Position],
    symbol_upper: str,
) -> list[Position]:
    return [p for p in positions if _is_equity_long(p, symbol_upper)]


def _aggregate_position_metrics(
    positions: list[Position],
    account: SchwabAccounts | None,
) -> tuple[float | None, float | None]:
    if not positions:
        return None, None

    liquidation = 0.0
    if account is not None:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue

    total_mv = sum(abs(p.marketValue) for p in positions if p.longQuantity > 0)
    weight_pct = None
    if liquidation > 0:
        weight_pct = (total_mv / liquidation) * 100.0
    elif positions[0].portfolioWeightPct is not None:
        weight_pct = positions[0].portfolioWeightPct

    pnl_values: list[float] = []
    for position in positions:
        if position.longQuantity <= 0:
            continue
        pnl = position.openProfitLossPct
        if pnl is None:
            pnl = position_open_profit_loss_pct(position)
        if pnl is not None:
            pnl_values.append(float(pnl))
    pnl_pct = min(pnl_values) if pnl_values else None
    return weight_pct, pnl_pct


def _pattern_extras(symbol_upper: str) -> dict[str, Any]:
    from app.services.pattern_intelligence_service import build_pattern_intelligence_payload

    out: dict[str, Any] = {
        "failed_breakout": False,
        "trend_bias": None,
        "rs_vs_spy_21d": None,
        "rs_vs_spy_63d": None,
    }
    try:
        pattern = build_pattern_intelligence_payload(symbol_upper, None)
    except Exception:
        return out
    if pattern is None:
        return out

    tc = pattern.trend_context
    out["trend_bias"] = tc.trend_bias
    out["rs_vs_spy_21d"] = tc.rs_vs_spy_21d
    out["rs_vs_spy_63d"] = tc.rs_vs_spy_63d
    chart_dict = (
        pattern.chart_intelligence.model_dump(by_alias=True)
        if hasattr(pattern.chart_intelligence, "model_dump")
        else dict(pattern.chart_intelligence or {})
    )
    _, _, _, failed = inputs_from_chart_payload(chart_dict)
    out["failed_breakout"] = failed
    return out


def _ranking_rank(symbol_upper: str) -> int | None:
    try:
        from ranking_pipeline.config import default_config
        from ranking_pipeline.storage.sqlite import open_store

        store = open_store(default_config())
        run_id = store.latest_run_id()
        if not run_id:
            return None
        row = store.get_symbol_ranking_row(run_id, symbol_upper)
        if row:
            return int(row["rank"])
    except Exception:
        logger.debug("Ranking rank lookup failed for %s", symbol_upper, exc_info=True)
    return None


def _symbol_alerts(
    *,
    portfolio_signals: list[IntelligenceSignal],
    proactive_alerts: list[ProactiveAlert],
    symbol_upper: str,
) -> list[str]:
    reasons: list[str] = []
    for alert in proactive_alerts:
        if alert.symbol and alert.symbol.upper() == symbol_upper:
            reasons.append(alert.reason)
    for signal in portfolio_signals:
        if signal.symbol and signal.symbol.upper() == symbol_upper:
            if signal.severity in {"warning", "critical"}:
                reasons.append(signal.message)
    return reasons


def build_equity_exit_guidance(
    *,
    symbol: str,
    positions: list[Position],
    account: SchwabAccounts | None,
    research_service: CompanyResearchService,
    proactive_alerts: list[ProactiveAlert] | None = None,
    all_positions: list[Position] | None = None,
) -> EquityExitGuidance:
    symbol_upper = symbol.strip().upper()
    scoped = _scoped_equity_positions(positions, symbol_upper)
    if not scoped:
        return EquityExitGuidance(
            symbol=symbol_upper,
            eligible=False,
            data_gaps=["no_equity_position"],
        )

    weight_pct, pnl_pct = _aggregate_position_metrics(scoped, account)
    trade = build_trade_decision(symbol_upper)
    extras = _pattern_extras(symbol_upper)
    rank = _ranking_rank(symbol_upper)

    research = research_service.build_context(symbol_upper)
    symbol_signals = SignalEngine.build_symbol_signals(
        research=research,
        positions=scoped,
        account=account,
        symbol=symbol_upper,
    )

    portfolio_signals: list[IntelligenceSignal] = []
    if account is not None and all_positions:
        portfolio_signals = SignalEngine.build_portfolio_signals(
            positions=all_positions,
            account=account,
        )

    alerts = proactive_alerts or []
    if not alerts and account is not None and all_positions:
        alerts = build_proactive_alerts(
            portfolio_signals=portfolio_signals,
            suggested_actions=[],
            earnings_this_week=[],
            assignment_risk_entries=None,
        )

    alert_reasons = _symbol_alerts(
        portfolio_signals=portfolio_signals,
        proactive_alerts=alerts,
        symbol_upper=symbol_upper,
    )

    result = evaluate_equity_exit_guidance(
        EquityExitGuidanceInputs(
            symbol=symbol_upper,
            as_of_date=trade.as_of_date,
            trade_decision=trade,
            signals=symbol_signals,
            alert_reasons=alert_reasons,
            position_weight_pct=weight_pct,
            open_profit_loss_pct=pnl_pct,
            failed_breakout=extras["failed_breakout"],
            trend_bias=extras["trend_bias"],
            rs_vs_spy_21d=extras["rs_vs_spy_21d"],
            rs_vs_spy_63d=extras["rs_vs_spy_63d"],
            ranking_rank=rank,
        )
    )

    data_gaps: list[str] = []
    if weight_pct is None:
        data_gaps.append("position_weight")
    if pnl_pct is None:
        data_gaps.append("open_pnl")

    return EquityExitGuidance(
        symbol=result.symbol,
        as_of_date=result.as_of_date,
        eligible=True,
        verdict=result.verdict,
        confidence=result.confidence,
        exit_urgency=result.exit_urgency,
        primary_reason=result.primary_reason,
        supporting_factors=result.supporting_factors,
        risk_factors=result.risk_factors,
        would_improve=result.would_improve,
        would_worsen=result.would_worsen,
        disclaimer=result.disclaimer,
        data_gaps=data_gaps,
        context=EquityExitGuidanceContext(
            regime_id=trade.regime.regime_id,
            trade_quality_score=trade.trade_quality_score,
            position_weight_pct=weight_pct,
            open_profit_loss_pct=pnl_pct,
            ranking_rank=rank,
        ),
    )


def build_portfolio_exit_attention(
    *,
    positions_by_symbol: dict[str, list[Position]],
    account: SchwabAccounts,
    research_service: CompanyResearchService,
    proactive_alerts: list[ProactiveAlert] | None = None,
    limit: int = 10,
) -> PortfolioExitAttentionResponse:
    all_positions = [p for plist in positions_by_symbol.values() for p in plist]
    items: list[PortfolioExitAttentionItem] = []

    for symbol, positions in positions_by_symbol.items():
        symbol_upper = symbol.upper()
        if not _scoped_equity_positions(positions, symbol_upper):
            continue
        try:
            guidance = build_equity_exit_guidance(
                symbol=symbol_upper,
                positions=positions,
                account=account,
                research_service=research_service,
                proactive_alerts=proactive_alerts,
                all_positions=all_positions,
            )
        except Exception:
            logger.exception("Exit guidance failed for %s", symbol_upper)
            continue
        if not guidance.eligible or guidance.verdict not in _ATTENTION_VERDICTS:
            continue
        if (
            guidance.verdict is None
            or guidance.confidence is None
            or guidance.exit_urgency is None
            or not guidance.primary_reason
        ):
            continue
        items.append(
            PortfolioExitAttentionItem(
                symbol=guidance.symbol,
                verdict=guidance.verdict,
                confidence=guidance.confidence,
                exit_urgency=guidance.exit_urgency,
                primary_reason=guidance.primary_reason,
            )
        )

    items.sort(key=lambda row: (-row.exit_urgency, row.symbol))
    return PortfolioExitAttentionResponse(items=items[: max(1, limit)])


def load_portfolio_for_user(
    *,
    user_id: str,
    portfolio_service: PortfolioService,
    access_token: str,
) -> dict[str, object]:
    del user_id
    return portfolio_service.get_enriched_account(access_token=access_token)
