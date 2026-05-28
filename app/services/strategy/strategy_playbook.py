from __future__ import annotations

from app.broker.portfolio_diversification import _aggregate_symbol_weights
from app.broker.strategy_symbol_alignment import strategy_symbol_list
from app.core.prompts import AnalysisAction
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import (
    InvestmentStrategy,
    StrategyNextAction,
    StrategySymbolStatus,
    UserInvestmentProfile,
    WheelPhase,
)

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)

WHEEL_PHASE_PRIORITY: dict[WheelPhase, int] = {
    WheelPhase.ASSIGNED_SHARES: 0,
    WheelPhase.SHORT_PUT_OPEN: 1,
    WheelPhase.SHORT_CALL_OPEN: 2,
    WheelPhase.READY_FOR_CSP: 3,
    WheelPhase.COMPLETE_CYCLE: 4,
    WheelPhase.PICK_SYMBOL: 5,
}

WHEEL_PHASE_LABELS: dict[WheelPhase, str] = {
    WheelPhase.PICK_SYMBOL: "Pick symbol",
    WheelPhase.READY_FOR_CSP: "Ready for CSP",
    WheelPhase.SHORT_PUT_OPEN: "Short put open",
    WheelPhase.ASSIGNED_SHARES: "Shares held",
    WheelPhase.SHORT_CALL_OPEN: "Short call open",
    WheelPhase.COMPLETE_CYCLE: "Cycle complete",
}


def _symbol_is_held(symbol: str, positions: list[Position]) -> bool:
    symbol_upper = symbol.upper()
    for position in positions:
        instrument = position.instrument
        underlying = (instrument.underlyingSymbol or instrument.symbol).upper()
        if underlying != symbol_upper:
            continue
        if instrument.assetType == "OPTION":
            if position.shortQuantity > 0 or position.longQuantity > 0:
                return True
        elif instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT"}:
            if position.longQuantity > 0 or position.shortQuantity > 0:
                return True
    return False


def _portfolio_weight_pct(
    symbol: str,
    *,
    positions: list[Position],
    account: SchwabAccounts | None,
) -> float | None:
    if account is None:
        return None
    liquidation = account.securitiesAccount.currentBalances.liquidationValue
    if not liquidation or liquidation <= 0:
        return None
    for ranked_symbol, _, weight in _aggregate_symbol_weights(positions, liquidation):
        if ranked_symbol.upper() == symbol.upper():
            return weight
    return None


def _wheel_status_label(
    *,
    strategy: InvestmentStrategy,
    phase: WheelPhase,
    held: bool,
) -> str:
    if strategy == InvestmentStrategy.CSP_INCOME:
        if phase == WheelPhase.READY_FOR_CSP:
            return "Ready to sell put"
        if phase == WheelPhase.SHORT_PUT_OPEN:
            return "Put open — monitor"
        if phase == WheelPhase.ASSIGNED_SHARES:
            return "Assigned — holding shares"
        if phase == WheelPhase.SHORT_CALL_OPEN:
            return "Call open — monitor"
    if strategy == InvestmentStrategy.COVERED_CALL:
        if phase == WheelPhase.READY_FOR_CSP and not held:
            return "Need shares for CC"
        if phase == WheelPhase.ASSIGNED_SHARES:
            return "Ready to write call"
        if phase == WheelPhase.SHORT_CALL_OPEN:
            return "Call open — monitor"
    return WHEEL_PHASE_LABELS.get(phase, phase.value)


def _dividend_status_label(*, held: bool) -> str:
    return "Held — review" if held else "Not held — research"


def _etf_status_label(*, held: bool, weight: float | None, target: float | None) -> str:
    if not held:
        return "Not held — add"
    if weight is not None and target is not None:
        drift = abs(weight - target)
        if drift > 3:
            return f"Held {weight:.1f}% — rebalance"
    return "Held — on target"


def next_action_for_symbol(
    *,
    strategy: InvestmentStrategy,
    symbol: str,
    wheel_phase: WheelPhase | None = None,
    held: bool = False,
    csp_candidates: list[dict] | None = None,
    covered_call_candidates: list[dict] | None = None,
) -> StrategyNextAction | None:
    if strategy in WHEEL_LIKE and wheel_phase is not None:
        if wheel_phase == WheelPhase.READY_FOR_CSP:
            if strategy == InvestmentStrategy.COVERED_CALL:
                return StrategyNextAction(
                    type="buy",
                    title=f"Buy {symbol} before writing calls",
                    reason="You need at least 100 shares to sell a covered call.",
                    symbol=symbol,
                )
            return StrategyNextAction(
                type="research",
                title=f"Research {symbol} before selling a put",
                reason="Confirm you'd be happy to own shares near your chosen strike.",
                symbol=symbol,
                action_id=AnalysisAction.RISK_CHECK.value,
            )
        if wheel_phase == WheelPhase.SHORT_PUT_OPEN:
            return StrategyNextAction(
                type="monitor",
                title=f"Monitor your {symbol} short put",
                reason="Watch delta and days to expiration; roll or hold before assignment.",
                symbol=symbol,
                action_id=AnalysisAction.ASSIGNMENT_RISK.value,
            )
        if wheel_phase == WheelPhase.ASSIGNED_SHARES:
            if strategy == InvestmentStrategy.CSP_INCOME:
                return StrategyNextAction(
                    type="monitor",
                    title=f"Monitor assigned {symbol} shares",
                    reason="Shares were assigned — decide whether to hold, sell, or write calls.",
                    symbol=symbol,
                )
            action = StrategyNextAction(
                type="options",
                title=f"Sell a covered call on {symbol}",
                reason="You hold shares — the next wheel leg is selling an out-of-the-money call.",
                symbol=symbol,
            )
            if covered_call_candidates:
                top = covered_call_candidates[0]
                return StrategyNextAction(
                    type="options",
                    title=f"Covered call candidate on {symbol}",
                    reason=top.get("rationale")
                    or "Top covered call candidate from your option chain.",
                    symbol=symbol,
                    metadata=top,
                )
            return action
        if wheel_phase == WheelPhase.SHORT_CALL_OPEN:
            return StrategyNextAction(
                type="monitor",
                title=f"Monitor {symbol} call assignment risk",
                reason="If called away, you're ready to sell another cash-secured put.",
                symbol=symbol,
                action_id=AnalysisAction.ASSIGNMENT_RISK.value,
            )
        if wheel_phase == WheelPhase.READY_FOR_CSP and csp_candidates:
            top = csp_candidates[0]
            return StrategyNextAction(
                type="options",
                title=f"Consider CSP on {symbol}",
                reason=top.get("rationale")
                or "Top cash-secured put candidate from your option chain.",
                symbol=symbol,
                metadata=top,
            )
        return None

    if strategy == InvestmentStrategy.DIVIDEND:
        if held:
            return StrategyNextAction(
                type="research",
                title=f"Review fundamentals for {symbol}",
                reason="Check yield, payout ratio, and cash flow before adding size.",
                symbol=symbol,
                action_id=AnalysisAction.RISK_CHECK.value,
            )
        return StrategyNextAction(
            type="research",
            title=f"Research {symbol} before buying",
            reason="Review dividend safety, yield, and valuation.",
            symbol=symbol,
            action_id=AnalysisAction.RISK_CHECK.value,
        )

    if strategy == InvestmentStrategy.ETF_CORE:
        if held:
            return StrategyNextAction(
                type="rebalance",
                title=f"Review {symbol} allocation",
                reason="Compare current weight to your target mix.",
                symbol=symbol,
            )
        return StrategyNextAction(
            type="buy",
            title=f"Add to {symbol}",
            reason="Start or continue building your core ETF position.",
            symbol=symbol,
        )

    return None


def build_symbol_statuses(
    *,
    profile: UserInvestmentProfile,
    strategy: InvestmentStrategy,
    positions: list[Position],
    account: SchwabAccounts | None,
    csp_candidates: list[dict] | None = None,
    covered_call_candidates: list[dict] | None = None,
    focus_symbol: str | None = None,
) -> list[StrategySymbolStatus]:
    symbols = strategy_symbol_list(profile)
    if not symbols:
        return []

    allocation = (
        profile.etf_core.target_allocation if profile.etf_core else {}
    )
    statuses: list[StrategySymbolStatus] = []

    for symbol in symbols:
        upper = symbol.upper()
        held = _symbol_is_held(upper, positions)
        weight = _portfolio_weight_pct(upper, positions=positions, account=account)
        wheel_phase = None
        status_label = "On your playbook"
        priority = 50
        action: StrategyNextAction | None = None

        if strategy in WHEEL_LIKE:
            from app.services.strategy.strategy_journey_service import (
                StrategyJourneyService,
            )

            wheel_phase = StrategyJourneyService.detect_wheel_phase(
                symbol=upper,
                positions=positions,
            )
            status_label = _wheel_status_label(
                strategy=strategy,
                phase=wheel_phase,
                held=held,
            )
            priority = WHEEL_PHASE_PRIORITY.get(wheel_phase, 50)
            use_csp = csp_candidates if upper == (focus_symbol or "").upper() else None
            use_cc = (
                covered_call_candidates
                if upper == (focus_symbol or "").upper()
                else None
            )
            action = next_action_for_symbol(
                strategy=strategy,
                symbol=upper,
                wheel_phase=wheel_phase,
                held=held,
                csp_candidates=use_csp,
                covered_call_candidates=use_cc,
            )
        elif strategy == InvestmentStrategy.DIVIDEND:
            status_label = _dividend_status_label(held=held)
            priority = 10 if not held else 20
            action = next_action_for_symbol(
                strategy=strategy,
                symbol=upper,
                held=held,
            )
        elif strategy == InvestmentStrategy.ETF_CORE:
            target = allocation.get(symbol) or allocation.get(upper)
            status_label = _etf_status_label(
                held=held,
                weight=weight,
                target=target,
            )
            priority = 10 if not held else 20
            action = next_action_for_symbol(
                strategy=strategy,
                symbol=upper,
                held=held,
            )

        if not held and strategy in WHEEL_LIKE:
            priority = min(priority, WHEEL_PHASE_PRIORITY[WheelPhase.READY_FOR_CSP])

        statuses.append(
            StrategySymbolStatus(
                symbol=upper,
                held=held,
                portfolio_weight_pct=weight,
                wheel_phase=wheel_phase,
                status_label=status_label,
                next_action=action,
                priority=priority,
            )
        )

    statuses.sort(key=lambda item: (item.priority, item.symbol))
    return statuses


def pick_focus_symbol(statuses: list[StrategySymbolStatus]) -> str | None:
    for status in statuses:
        if status.next_action is not None:
            return status.symbol
    if statuses:
        return statuses[0].symbol
    return None
