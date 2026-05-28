from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.adapters.user.user_investment_profile_adapter import UserInvestmentProfileAdapter
from app.adapters.user.user_strategy_journey_adapter import UserStrategyJourneyAdapter
from app.broker.strategy_detector import SHARES_PER_OPTION_CONTRACT, detect_option_strategy
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStep,
    JourneyStepStatus,
    JourneyStepUpdate,
    StrategyCatalogItem,
    StrategyNextAction,
    StrategyReadiness,
    StrategyRecommendations,
    UserInvestmentProfile,
    UserInvestmentProfileUpdate,
    UserStrategyJourney,
    WheelPhase,
)
from app.services.strategy.strategy_catalog import (
    STRATEGY_CATALOG,
    build_initial_steps,
    catalog_item,
)
from app.services.strategy.strategy_playbook import (
    build_symbol_statuses,
    pick_focus_symbol,
)

logger = logging.getLogger(__name__)

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)


class StrategyJourneyService:
    def __init__(
        self,
        profile_adapter: UserInvestmentProfileAdapter,
        journey_adapter: UserStrategyJourneyAdapter,
    ):
        self.profile_adapter = profile_adapter
        self.journey_adapter = journey_adapter

    def list_catalog(self) -> list[StrategyCatalogItem]:
        return list(STRATEGY_CATALOG)

    def get_profile(self, *, user_id: str) -> UserInvestmentProfile | None:
        try:
            return self.profile_adapter.get_by_user_id(user_id)
        except Exception:
            logger.exception("Failed to load investment profile for %s", user_id)
            return None

    def upsert_profile(
        self,
        *,
        user_id: str,
        update: UserInvestmentProfileUpdate,
    ) -> UserInvestmentProfile:
        existing = self.get_profile(user_id=user_id)
        return self.profile_adapter.upsert(
            user_id=user_id,
            update=update,
            existing=existing,
        )

    def select_strategy(
        self,
        *,
        user_id: str,
        strategy: InvestmentStrategy,
    ) -> tuple[UserInvestmentProfile, UserStrategyJourney]:
        profile = self.upsert_profile(
            user_id=user_id,
            update=UserInvestmentProfileUpdate(primary_strategy=strategy),
        )
        steps = build_initial_steps(strategy)
        journey = self.journey_adapter.upsert(
            user_id=user_id,
            strategy=strategy,
            steps=steps,
            current_step_id=steps[0].step_id if steps else None,
        )
        return profile, journey

    def get_journey(
        self,
        *,
        user_id: str,
        strategy: InvestmentStrategy | None = None,
    ) -> UserStrategyJourney | None:
        profile = self.get_profile(user_id=user_id)
        target = strategy or (profile.primary_strategy if profile else None)
        if target is None:
            return None
        try:
            return self.journey_adapter.get_by_user_and_strategy(user_id, target)
        except Exception:
            logger.exception(
                "Failed to load strategy journey for %s/%s", user_id, target.value
            )
            return None

    def update_step(
        self,
        *,
        user_id: str,
        strategy: InvestmentStrategy,
        step_id: str,
        update: JourneyStepUpdate,
    ) -> UserStrategyJourney | None:
        journey = self.get_journey(user_id=user_id, strategy=strategy)
        if journey is None:
            return None

        steps = list(journey.steps)
        target_index = next(
            (index for index, step in enumerate(steps) if step.step_id == step_id),
            None,
        )
        if target_index is None:
            return None

        completed_at = (
            datetime.now(timezone.utc)
            if update.status in {JourneyStepStatus.COMPLETED, JourneyStepStatus.SKIPPED}
            else None
        )
        steps[target_index] = steps[target_index].model_copy(
            update={
                "status": update.status,
                "completed_at": completed_at,
                "metadata": update.metadata or steps[target_index].metadata,
            }
        )

        if update.status == JourneyStepStatus.COMPLETED and target_index + 1 < len(steps):
            next_step = steps[target_index + 1]
            if next_step.status == JourneyStepStatus.LOCKED:
                steps[target_index + 1] = next_step.model_copy(
                    update={"status": JourneyStepStatus.AVAILABLE}
                )

        current_step_id = self._resolve_current_step_id(steps)
        all_done = all(
            step.status in {JourneyStepStatus.COMPLETED, JourneyStepStatus.SKIPPED}
            for step in steps
        )
        completed_at_journey = datetime.now(timezone.utc) if all_done else None

        return self.journey_adapter.upsert(
            user_id=user_id,
            strategy=strategy,
            steps=steps,
            current_step_id=current_step_id,
            completed_at=completed_at_journey,
        )

    def sync_journey_progress(
        self,
        *,
        user_id: str,
        schwab_linked: bool,
        positions: list[Position] | None = None,
        account: SchwabAccounts | None = None,
        recent_option_activity: bool = False,
    ) -> UserStrategyJourney | None:
        profile = self.get_profile(user_id=user_id)
        if profile is None or profile.primary_strategy is None:
            return None

        journey = self.get_journey(user_id=user_id, strategy=profile.primary_strategy)
        if journey is None:
            return None

        steps = list(journey.steps)
        changed = False

        def complete(step_id: str, metadata: dict | None = None) -> None:
            nonlocal changed
            index = next(
                (idx for idx, step in enumerate(steps) if step.step_id == step_id),
                None,
            )
            if index is None:
                return
            step = steps[index]
            if step.status in {JourneyStepStatus.COMPLETED, JourneyStepStatus.SKIPPED}:
                return
            steps[index] = step.model_copy(
                update={
                    "status": JourneyStepStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "metadata": {**step.metadata, **(metadata or {})},
                }
            )
            if index + 1 < len(steps) and steps[index + 1].status == JourneyStepStatus.LOCKED:
                steps[index + 1] = steps[index + 1].model_copy(
                    update={"status": JourneyStepStatus.AVAILABLE}
                )
            changed = True

        if schwab_linked:
            complete("connect-schwab")

        strategy = profile.primary_strategy
        if strategy in WHEEL_LIKE and profile.wheel and profile.wheel.wheel_symbols:
            complete(
                "pick-underlying",
                {"symbols": profile.wheel.wheel_symbols},
            )
        if strategy == InvestmentStrategy.DIVIDEND and profile.dividend:
            if profile.dividend.dividend_symbols:
                complete(
                    "pick-dividend-names",
                    {"symbols": profile.dividend.dividend_symbols},
                )
            if profile.income_vs_growth or profile.risk_tolerance:
                complete("set-income-preferences")
        if strategy == InvestmentStrategy.ETF_CORE and profile.etf_core:
            if profile.etf_core.target_allocation:
                complete(
                    "set-allocation",
                    {"allocation": profile.etf_core.target_allocation},
                )

        positions = positions or []
        if positions:
            if strategy == InvestmentStrategy.COVERED_CALL:
                if self._has_covered_call(positions):
                    complete("sell-first-call")
                if self._has_share_lot(positions):
                    complete("confirm-share-count")

            wheel_phase = self.detect_wheel_phase(
                symbol=self._primary_symbol(profile, positions),
                positions=positions,
            )
            if wheel_phase in {
                WheelPhase.SHORT_PUT_OPEN,
                WheelPhase.ASSIGNED_SHARES,
                WheelPhase.SHORT_CALL_OPEN,
                WheelPhase.COMPLETE_CYCLE,
            }:
                complete("sell-first-csp")
            if wheel_phase in {
                WheelPhase.SHORT_PUT_OPEN,
                WheelPhase.ASSIGNED_SHARES,
                WheelPhase.SHORT_CALL_OPEN,
            }:
                complete("monitor-or-roll")
            if wheel_phase in {WheelPhase.SHORT_CALL_OPEN, WheelPhase.COMPLETE_CYCLE}:
                complete("sell-covered-call")
            if wheel_phase == WheelPhase.COMPLETE_CYCLE:
                complete("complete-cycle")

            if strategy == InvestmentStrategy.DIVIDEND and self._has_equity_positions(positions):
                complete("first-dividend-buy")
            if strategy == InvestmentStrategy.ETF_CORE and self._has_equity_positions(positions):
                complete("first-etf-buy")

        if recent_option_activity and strategy in WHEEL_LIKE:
            complete("sell-first-csp")

        if not changed:
            return journey

        current_step_id = self._resolve_current_step_id(steps)
        all_done = all(
            step.status in {JourneyStepStatus.COMPLETED, JourneyStepStatus.SKIPPED}
            for step in steps
        )
        return self.journey_adapter.upsert(
            user_id=user_id,
            strategy=strategy,
            steps=steps,
            current_step_id=current_step_id,
            completed_at=datetime.now(timezone.utc) if all_done else None,
        )

    def build_recommendations(
        self,
        *,
        user_id: str,
        strategy: InvestmentStrategy | None = None,
        symbol: str | None = None,
        schwab_linked: bool,
        positions: list[Position] | None = None,
        account: SchwabAccounts | None = None,
        csp_candidates: list[dict] | None = None,
        covered_call_candidates: list[dict] | None = None,
    ) -> StrategyRecommendations | None:
        profile = self.get_profile(user_id=user_id)
        target = strategy or (profile.primary_strategy if profile else None)
        if target is None or profile is None:
            return None

        journey = self.sync_journey_progress(
            user_id=user_id,
            schwab_linked=schwab_linked,
            positions=positions,
            account=account,
        )
        current_step = self._current_step(journey)

        readiness = self._build_readiness(
            profile=profile,
            schwab_linked=schwab_linked,
            positions=positions or [],
            account=account,
        )

        positions = positions or []
        symbol_statuses = build_symbol_statuses(
            profile=profile,
            strategy=target,
            positions=positions,
            account=account,
            csp_candidates=csp_candidates,
            covered_call_candidates=covered_call_candidates,
            focus_symbol=symbol.upper() if symbol else None,
        )
        focus_symbol = (
            symbol.upper()
            if symbol
            else pick_focus_symbol(symbol_statuses)
            or self._primary_symbol(profile, positions)
        )
        if focus_symbol and symbol_statuses:
            symbol_statuses = build_symbol_statuses(
                profile=profile,
                strategy=target,
                positions=positions,
                account=account,
                csp_candidates=csp_candidates,
                covered_call_candidates=covered_call_candidates,
                focus_symbol=focus_symbol,
            )

        wheel_phase = None
        if target in WHEEL_LIKE and focus_symbol:
            wheel_phase = self.detect_wheel_phase(
                symbol=focus_symbol,
                positions=positions,
            )

        next_actions: list[StrategyNextAction] = []

        if not schwab_linked:
            next_actions.append(
                StrategyNextAction(
                    type="connect",
                    title="Connect Schwab",
                    reason="Link your brokerage account to track progress and get personalized suggestions.",
                )
            )
            return StrategyRecommendations(
                strategy=target,
                current_step=current_step,
                wheel_phase=wheel_phase,
                readiness=readiness,
                symbol=focus_symbol,
                symbol_statuses=symbol_statuses,
                next_actions=next_actions,
            )

        if target in WHEEL_LIKE and profile.wheel and not profile.wheel.wheel_symbols:
            next_actions.append(
                StrategyNextAction(
                    type="education",
                    title="Pick a wheel underlying",
                    reason="Choose a stock you're comfortable owning if assigned on a cash-secured put.",
                )
            )
        elif target == InvestmentStrategy.DIVIDEND and profile.dividend and not profile.dividend.dividend_symbols:
            next_actions.append(
                StrategyNextAction(
                    type="education",
                    title="Pick dividend names",
                    reason="Choose 3–5 reliable payers to research and hold.",
                )
            )
        elif target == InvestmentStrategy.ETF_CORE and profile.etf_core and not profile.etf_core.target_allocation:
            next_actions.append(
                StrategyNextAction(
                    type="education",
                    title="Set your target allocation",
                    reason="Define your broad market / bond mix before buying.",
                )
            )
        else:
            for status in symbol_statuses:
                if status.next_action is not None:
                    next_actions.append(status.next_action)

        if target == InvestmentStrategy.ETF_CORE and profile.etf_core and profile.etf_core.target_allocation:
            next_actions.append(
                StrategyNextAction(
                    type="rebalance",
                    title="Review allocation drift",
                    reason="Compare current weights to your target mix.",
                )
            )

        if not next_actions and current_step:
            next_actions.append(
                StrategyNextAction(
                    type="education",
                    title=current_step.title,
                    reason=current_step.description,
                )
            )

        return StrategyRecommendations(
            strategy=target,
            current_step=current_step,
            wheel_phase=wheel_phase,
            readiness=readiness,
            symbol=focus_symbol,
            symbol_statuses=symbol_statuses,
            next_actions=next_actions,
        )

    @staticmethod
    def detect_wheel_phase(
        *,
        symbol: str | None,
        positions: list[Position],
    ) -> WheelPhase:
        if not symbol:
            return WheelPhase.PICK_SYMBOL

        symbol_upper = symbol.upper()
        share_qty = 0.0
        short_put = False
        short_call = False

        for position in positions:
            instrument = position.instrument
            underlying = (
                instrument.underlyingSymbol or instrument.symbol
            ).upper()
            if underlying != symbol_upper:
                continue

            if instrument.assetType == "OPTION":
                strategy = detect_option_strategy(position, positions)
                if strategy == "cash_secured_put" and position.shortQuantity > 0:
                    short_put = True
                if strategy == "covered_call" and position.shortQuantity > 0:
                    short_call = True
            elif instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT"}:
                share_qty += position.longQuantity - position.shortQuantity

        if short_call:
            return WheelPhase.SHORT_CALL_OPEN
        if share_qty >= SHARES_PER_OPTION_CONTRACT and not short_put:
            return WheelPhase.ASSIGNED_SHARES
        if short_put:
            return WheelPhase.SHORT_PUT_OPEN
        if share_qty > 0:
            return WheelPhase.ASSIGNED_SHARES
        return WheelPhase.READY_FOR_CSP

    @staticmethod
    def _resolve_current_step_id(steps: list[JourneyStep]) -> str | None:
        for step in steps:
            if step.status in {
                JourneyStepStatus.AVAILABLE,
                JourneyStepStatus.IN_PROGRESS,
            }:
                return step.step_id
        for step in steps:
            if step.status == JourneyStepStatus.LOCKED:
                return step.step_id
        return steps[-1].step_id if steps else None

    @staticmethod
    def _current_step(journey: UserStrategyJourney | None) -> JourneyStep | None:
        if journey is None:
            return None
        if journey.current_step_id:
            for step in journey.steps:
                if step.step_id == journey.current_step_id:
                    return step
        for step in journey.steps:
            if step.status in {
                JourneyStepStatus.AVAILABLE,
                JourneyStepStatus.IN_PROGRESS,
            }:
                return step
        return journey.steps[-1] if journey.steps else None

    @staticmethod
    def _primary_symbol(
        profile: UserInvestmentProfile,
        positions: list[Position],
    ) -> str | None:
        if profile.wheel and profile.wheel.wheel_symbols:
            return profile.wheel.wheel_symbols[0].upper()
        if profile.dividend and profile.dividend.dividend_symbols:
            return profile.dividend.dividend_symbols[0].upper()
        if profile.etf_core and profile.etf_core.target_allocation:
            return next(iter(profile.etf_core.target_allocation.keys())).upper()
        for position in positions:
            instrument = position.instrument
            if instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT"}:
                return instrument.symbol.upper()
        return None

    @staticmethod
    def _build_readiness(
        *,
        profile: UserInvestmentProfile,
        schwab_linked: bool,
        positions: list[Position],
        account: SchwabAccounts | None,
    ) -> StrategyReadiness:
        cash = None
        if account is not None:
            cash = account.securitiesAccount.currentBalances.cashBalance

        approved: list[str] = []
        if profile.wheel and profile.wheel.wheel_symbols:
            approved.extend(sym.upper() for sym in profile.wheel.wheel_symbols)
        if profile.dividend and profile.dividend.dividend_symbols:
            approved.extend(sym.upper() for sym in profile.dividend.dividend_symbols)
        if profile.etf_core and profile.etf_core.target_allocation:
            approved.extend(sym.upper() for sym in profile.etf_core.target_allocation)

        return StrategyReadiness(
            schwab_linked=schwab_linked,
            has_positions=len(positions) > 0,
            cash_available=cash,
            approved_symbols=sorted(set(approved)),
        )

    @staticmethod
    def _has_covered_call(positions: list[Position]) -> bool:
        for position in positions:
            if detect_option_strategy(position, positions) == "covered_call":
                return True
        return False

    @staticmethod
    def _has_share_lot(positions: list[Position]) -> bool:
        for position in positions:
            if position.instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT"}:
                if position.longQuantity >= SHARES_PER_OPTION_CONTRACT:
                    return True
        return False

    @staticmethod
    def _has_equity_positions(positions: list[Position]) -> bool:
        return any(
            position.instrument.assetType in {"EQUITY", "COLLECTIVE_INVESTMENT"}
            for position in positions
        )

    @staticmethod
    def get_catalog_item(strategy: InvestmentStrategy) -> StrategyCatalogItem:
        return catalog_item(strategy)
