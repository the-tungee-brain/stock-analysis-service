from __future__ import annotations

import logging
from typing import Any

from app.broker.option_utils import (
    classify_moneyness,
    assignment_risk_level,
    is_short_option,
)
from app.builders.equity_exit_guidance_engine import (
    EquityExitGuidanceInputs,
    evaluate_equity_exit_guidance,
)
from app.builders.option_position_guidance_engine import (
    LongOptionGuidanceInputs,
    ShortOptionGuidanceInputs,
    evaluate_long_option,
    evaluate_short_option,
)
from app.builders.position_guidance_support import (
    account_liquidation_value,
    format_display_label,
    position_key,
    position_metrics,
    position_quantity,
    positions_for_symbol,
)
from app.builders.guidance_scoring_types import GuidanceDriver, justification_label
from app.builders.position_guidance_relative_risk import compute_relative_risk_rank
from app.builders.symbol_thesis_engine import evaluate_symbol_thesis
from app.builders.trade_decision_engine import inputs_from_chart_payload
from app.models.intelligence_models import IntelligenceSignal, ProactiveAlert
from app.models.position_guidance_models import (
    GuidanceDriverModel,
    PortfolioExitAttentionItem,
    PortfolioExitAttentionResponse,
    PositionGuidanceItem,
    PositionKind,
    PositionVerdict,
    SymbolPositionGuidanceResponse,
    SymbolThesisBlock,
)
from app.models.schwab_models import Position, SchwabAccounts
from app.services.company_research_service import CompanyResearchService
from app.services.intelligence.signal_engine import SignalEngine, build_proactive_alerts
from app.services.portfolio_service import PortfolioService
from app.services.trade_decision_service import build_trade_decision

logger = logging.getLogger(__name__)

_ATTENTION_VERDICTS: frozenset[PositionVerdict] = frozenset({
    "TRIM",
    "REVIEW_SELL",
    "EXIT",
    "REVIEW_CLOSE",
    "CLOSE",
    "ROLL",
    "REVIEW_ASSIGNMENT_RISK",
})

def _driver_model(driver: GuidanceDriver) -> GuidanceDriverModel:
    return GuidanceDriverModel(
        code=driver.code,
        label=driver.label,
        points=driver.points,
        detail=driver.detail,
    )


_VERDICT_LABELS: dict[str, str] = {
    "HOLD": "Hold",
    "TRIM": "Trim",
    "REVIEW_SELL": "Review sell",
    "EXIT": "Exit",
    "REVIEW_CLOSE": "Review close",
    "CLOSE": "Close",
    "ROLL": "Roll",
    "REVIEW_ASSIGNMENT_RISK": "Review assignment risk",
}


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


def _infer_underlying_price(positions: list[Position], symbol_upper: str) -> float | None:
    for position in positions:
        if position.longQuantity <= 0:
            continue
        instrument = position.instrument
        if (instrument.assetType or "").upper() == "OPTION":
            continue
        if (instrument.symbol or "").upper() != symbol_upper:
            continue
        if position.longQuantity > 0 and position.marketValue:
            return abs(position.marketValue) / position.longQuantity
    return None


def _build_synthesis(
    *,
    symbol_upper: str,
    thesis_block: SymbolThesisBlock,
    position_items: list[PositionGuidanceItem],
) -> tuple[str, str]:
    thesis_line = (
        f"Symbol thesis for {symbol_upper} is {thesis_block.thesis.title()} "
        f"({thesis_block.summary})"
    )
    if not position_items:
        narrative = thesis_line
        prompt = (
            f"Explain the {symbol_upper} thesis and what it means for how I should "
            "think about new positions."
        )
        return narrative, prompt

    position_lines = [
        f"{item.display_label}: {_VERDICT_LABELS.get(item.verdict, item.verdict)} "
        f"— {item.primary_reason}"
        for item in position_items
    ]
    narrative = thesis_line + " " + " ".join(position_lines)
    prompt = (
        f"Explain my {symbol_upper} position-level guidance and synthesize how these "
        f"positions fit together. Use this context: {narrative}"
    )
    return narrative, prompt


def build_symbol_position_guidance(
    *,
    symbol: str,
    positions: list[Position],
    account: SchwabAccounts | None,
    research_service: CompanyResearchService,
    proactive_alerts: list[ProactiveAlert] | None = None,
    all_positions: list[Position] | None = None,
) -> SymbolPositionGuidanceResponse:
    symbol_upper = symbol.strip().upper()
    scoped = positions_for_symbol(positions, symbol_upper)
    trade = build_trade_decision(symbol_upper)
    extras = _pattern_extras(symbol_upper)
    thesis_result = evaluate_symbol_thesis(trade, trend_bias=extras["trend_bias"])
    thesis_block = SymbolThesisBlock(
        thesis=thesis_result.thesis,
        summary=thesis_result.summary,
        trade_quality_score=trade.trade_quality_score,
        regime_id=trade.regime.regime_id,
    )

    if not scoped:
        narrative, prompt = _build_synthesis(
            symbol_upper=symbol_upper,
            thesis_block=thesis_block,
            position_items=[],
        )
        return SymbolPositionGuidanceResponse(
            symbol=symbol_upper,
            as_of_date=trade.as_of_date,
            has_positions=False,
            thesis=thesis_block,
            synthesis_narrative=narrative,
            analysis_prompt=prompt,
        )

    liquidation = account_liquidation_value(account)
    rank = _ranking_rank(symbol_upper)
    underlying_price = _infer_underlying_price(positions, symbol_upper)

    research = research_service.build_context(symbol_upper)
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

    symbol_signals = SignalEngine.build_symbol_signals(
        research=research,
        positions=[p for p, _ in scoped],
        account=account,
        symbol=symbol_upper,
    )

    items: list[PositionGuidanceItem] = []
    data_gaps: list[str] = []

    for position, kind in scoped:
        metrics = position_metrics(position, account_liquidation=liquidation)
        label = format_display_label(
            position_kind=kind,
            position=position,
            metrics=metrics,
        )
        key = position_key(position)
        qty = position_quantity(position)

        if kind == "EQUITY_LONG":
            equity_result = evaluate_equity_exit_guidance(
                EquityExitGuidanceInputs(
                    symbol=symbol_upper,
                    as_of_date=trade.as_of_date,
                    trade_decision=trade,
                    signals=symbol_signals,
                    alert_reasons=alert_reasons,
                    position_weight_pct=metrics.weight_pct,
                    open_profit_loss_pct=metrics.pnl_pct,
                    failed_breakout=extras["failed_breakout"],
                    trend_bias=extras["trend_bias"],
                    rs_vs_spy_21d=extras["rs_vs_spy_21d"],
                    rs_vs_spy_63d=extras["rs_vs_spy_63d"],
                    ranking_rank=rank,
                    position_quantity=qty,
                )
            )
            rel_rank = compute_relative_risk_rank(
                position_kind=kind,
                verdict=equity_result.verdict,
                urgency=equity_result.exit_urgency,
                open_profit_loss_pct=metrics.pnl_pct,
                position_weight_pct=metrics.weight_pct,
            )
            items.append(
                PositionGuidanceItem(
                    position_key=key,
                    position_kind=kind,
                    display_label=label,
                    instrument_symbol=position.instrument.symbol.upper(),
                    underlying_symbol=symbol_upper,
                    quantity=qty,
                    market_value=position.marketValue,
                    open_profit_loss_pct=metrics.pnl_pct,
                    verdict=equity_result.verdict,
                    confidence=equity_result.confidence,
                    urgency=equity_result.exit_urgency,
                    relative_risk_rank=rel_rank,
                    justification=justification_label(equity_result.justification),
                    primary_driver=_driver_model(equity_result.primary_driver),
                    secondary_driver=(
                        _driver_model(equity_result.secondary_driver)
                        if equity_result.secondary_driver
                        else None
                    ),
                    tertiary_driver=(
                        _driver_model(equity_result.tertiary_driver)
                        if equity_result.tertiary_driver
                        else None
                    ),
                    primary_reason=equity_result.primary_reason,
                    supporting_factors=equity_result.supporting_factors,
                    risk_factors=equity_result.risk_factors,
                )
            )
            if metrics.weight_pct is None:
                data_gaps.append(f"{key}:position_weight")
            if metrics.pnl_pct is None:
                data_gaps.append(f"{key}:open_pnl")
            continue

        put_call = position.instrument.putCall
        moneyness = "unknown"
        if kind.startswith("SHORT") or kind.startswith("LONG"):
            strike = metrics.strike
            if strike is not None and put_call:
                moneyness = classify_moneyness(
                    put_call=put_call,
                    strike=strike,
                    underlying_price=underlying_price,
                )

        if kind in {"LONG_CALL", "LONG_PUT"}:
            opt = evaluate_long_option(
                LongOptionGuidanceInputs(
                    position_kind=kind,
                    thesis=thesis_block.thesis,
                    dte=metrics.dte,
                    pnl_pct=metrics.pnl_pct,
                    moneyness=moneyness,
                    alert_reasons=alert_reasons,
                )
            )
            items.append(
                _option_item(
                    position=position,
                    kind=kind,
                    key=key,
                    label=label,
                    symbol_upper=symbol_upper,
                    qty=qty,
                    metrics=metrics,
                    put_call=put_call,
                    result=opt,
                )
            )
        else:
            risk_level = assignment_risk_level(
                moneyness=moneyness,
                days_to_expiry=metrics.dte if metrics.dte is not None else 999,
            )
            opt = evaluate_short_option(
                ShortOptionGuidanceInputs(
                    position_kind=kind,
                    thesis=thesis_block.thesis,
                    dte=metrics.dte,
                    pnl_pct=metrics.pnl_pct,
                    moneyness=moneyness,
                    assignment_risk=risk_level,
                    option_strategy=position.optionStrategy,
                    alert_reasons=alert_reasons,
                )
            )
            items.append(
                _option_item(
                    position=position,
                    kind=kind,
                    key=key,
                    label=label,
                    symbol_upper=symbol_upper,
                    qty=qty,
                    metrics=metrics,
                    put_call=put_call,
                    result=opt,
                )
            )

        if metrics.dte is None:
            data_gaps.append(f"{key}:expiration")
        if underlying_price is None and is_short_option(position):
            data_gaps.append(f"{key}:underlying_price")

    items.sort(key=lambda row: (-row.relative_risk_rank, -row.urgency, row.display_label))
    narrative, prompt = _build_synthesis(
        symbol_upper=symbol_upper,
        thesis_block=thesis_block,
        position_items=items,
    )

    return SymbolPositionGuidanceResponse(
        symbol=symbol_upper,
        as_of_date=trade.as_of_date,
        has_positions=True,
        thesis=thesis_block,
        positions=items,
        synthesis_narrative=narrative,
        analysis_prompt=prompt,
        data_gaps=data_gaps,
    )


def _option_item(
    *,
    position: Position,
    kind: PositionKind,
    key: str,
    label: str,
    symbol_upper: str,
    qty: float,
    metrics,
    put_call: str | None,
    result,
) -> PositionGuidanceItem:
    rel_rank = compute_relative_risk_rank(
        position_kind=kind,
        verdict=result.verdict,
        urgency=result.urgency,
        open_profit_loss_pct=metrics.pnl_pct,
    )
    return PositionGuidanceItem(
        position_key=key,
        position_kind=kind,
        display_label=label,
        instrument_symbol=(position.instrument.symbol or "").upper(),
        underlying_symbol=symbol_upper,
        put_call=put_call,
        strike=metrics.strike,
        expiration=metrics.expiration,
        quantity=qty,
        market_value=position.marketValue,
        open_profit_loss_pct=metrics.pnl_pct,
        verdict=result.verdict,
        confidence=result.confidence,
        urgency=result.urgency,
        relative_risk_rank=rel_rank,
        justification=justification_label(result.justification),
        primary_driver=_driver_model(result.primary_driver),
        secondary_driver=(
            _driver_model(result.secondary_driver) if result.secondary_driver else None
        ),
        tertiary_driver=(
            _driver_model(result.tertiary_driver) if result.tertiary_driver else None
        ),
        primary_reason=result.primary_reason,
        supporting_factors=result.supporting_factors,
        risk_factors=result.risk_factors,
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
        try:
            guidance = build_symbol_position_guidance(
                symbol=symbol_upper,
                positions=positions,
                account=account,
                research_service=research_service,
                proactive_alerts=proactive_alerts,
                all_positions=all_positions,
            )
        except Exception:
            logger.exception("Position guidance failed for %s", symbol_upper)
            continue

        for row in guidance.positions:
            if row.verdict not in _ATTENTION_VERDICTS:
                continue
            items.append(
                PortfolioExitAttentionItem(
                    position_key=row.position_key,
                    symbol=guidance.symbol,
                    position_kind=row.position_kind,
                    display_label=row.display_label,
                    verdict=row.verdict,
                    confidence=row.confidence,
                    urgency=row.urgency,
                    relative_risk_rank=row.relative_risk_rank,
                    primary_reason=row.primary_reason,
                )
            )

    items.sort(
        key=lambda row: (
            -row.relative_risk_rank,
            -row.urgency,
            row.symbol,
            row.display_label,
        )
    )
    return PortfolioExitAttentionResponse(items=items[: max(1, limit)])


def load_portfolio_for_user(
    *,
    user_id: str,
    portfolio_service: PortfolioService,
    access_token: str,
) -> dict[str, object]:
    del user_id
    return portfolio_service.get_enriched_account(access_token=access_token)
