from datetime import datetime, timezone
from typing import List, Optional

from app.adapters.cache.recent_orders_cache import RecentOrdersCache
from app.broker.order_grouping import (
    ActivityGroupInfo,
    detect_roll_groups,
    detect_wash_sale_flags,
    last_fill_time_for_symbol,
    leg_contract_label,
    leg_expiration,
    leg_put_call,
    leg_strike,
    spread_group_for_order,
)
from app.broker.order_utils import (
    is_order_within_days,
    leg_option_fields,
    order_asset_type,
    order_average_fill_price,
    order_fill_time,
    order_leg_average_fill_price,
    order_net_total_cash,
    order_premium_fields,
    order_primary_leg,
    order_relates_to_symbol,
    order_symbols,
    order_total_cash,
    order_underlying_symbol,
)
from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.core.prompts import AnalysisAction
from app.models.recent_order_models import (
    RecentActivitySummary,
    RecentActivitySymbolSummary,
    RecentOrderEntry,
    RecentOrderLegEntry,
    RecentOrdersResponse,
    SuggestedAnalysisAction,
)
from app.models.schwab_order_models import OrderLeg, SchwabOrder

DEFAULT_DAYS_BACK = 30
RECENT_ACTIVITY_DAYS = 7
WASH_SALE_WINDOW_DAYS = 30
PORTFOLIO_LATEST_ORDERS_LIMIT = 5
PORTFOLIO_SYMBOLS_LIMIT = 8
TAX_RELEVANT_SELL_INSTRUCTIONS = frozenset({"SELL", "SELL_TO_CLOSE"})
RISK_RELEVANT_BUY_INSTRUCTIONS = frozenset({"BUY", "BUY_TO_OPEN"})


class TransactionService:
    def __init__(
        self,
        schwab_trader_builder: SchwabTraderBuilder,
        recent_orders_cache: RecentOrdersCache | None = None,
    ):
        self.schwab_trader_builder = schwab_trader_builder
        self.recent_orders_cache = recent_orders_cache

    def invalidate_recent_orders_cache(self, *, user_id: str) -> int:
        if not self.recent_orders_cache:
            return 0
        return self.recent_orders_cache.invalidate_user(user_id=user_id)

    def _fetch_filled_orders(
        self,
        account_number: str,
        access_token: str,
        *,
        user_id: Optional[str] = None,
        days_back: int = DEFAULT_DAYS_BACK,
        refresh: bool = False,
    ) -> List[SchwabOrder]:
        if refresh and self.recent_orders_cache and user_id:
            self.recent_orders_cache.delete(
                user_id=user_id,
                account_number=account_number,
                days_back=days_back,
            )

        if not refresh and self.recent_orders_cache and user_id:
            cached = self.recent_orders_cache.get(
                user_id=user_id,
                account_number=account_number,
                days_back=days_back,
            )
            if cached is not None:
                return cached

        orders = self.schwab_trader_builder.get_orders(
            account_number=account_number,
            access_token=access_token,
            status="FILLED",
            days_back=days_back,
        )

        if self.recent_orders_cache and user_id:
            self.recent_orders_cache.put(
                user_id=user_id,
                account_number=account_number,
                days_back=days_back,
                orders=orders,
            )

        return orders

    def get_filled_orders(
        self,
        account_number: str,
        access_token: str,
        *,
        user_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days_back: int = DEFAULT_DAYS_BACK,
        refresh: bool = False,
    ) -> List[SchwabOrder]:
        orders = self._fetch_filled_orders(
            account_number=account_number,
            access_token=access_token,
            user_id=user_id,
            days_back=days_back,
            refresh=refresh,
        )

        if symbol is None:
            return orders

        target = symbol.upper()
        return [order for order in orders if order_relates_to_symbol(order, target)]

    def get_filled_orders_by_symbol(
        self,
        account_number: str,
        access_token: str,
        symbol: str,
        *,
        user_id: Optional[str] = None,
        days_back: int = DEFAULT_DAYS_BACK,
        refresh: bool = False,
    ) -> List[SchwabOrder]:
        return self.get_filled_orders(
            account_number=account_number,
            access_token=access_token,
            user_id=user_id,
            symbol=symbol,
            days_back=days_back,
            refresh=refresh,
        )

    def to_recent_order_leg_entry(
        self, order: SchwabOrder, leg: OrderLeg
    ) -> RecentOrderLegEntry:
        leg_id = leg.legId
        qty = leg.quantity if leg.quantity is not None else order.filledQuantity
        avg_fill = order_leg_average_fill_price(order, leg_id)
        premium_per_contract, total_premium = order_premium_fields(
            leg,
            fill_price_per_share=avg_fill,
            quantity=qty,
        )
        total_cash = order_total_cash(
            leg,
            fill_price_per_share=avg_fill,
            quantity=qty,
        )
        option_fields = leg_option_fields(leg)

        return RecentOrderLegEntry(
            leg_id=leg_id,
            instruction=(leg.instruction or "UNKNOWN").upper(),
            quantity=qty,
            asset_type=order_asset_type(leg),
            option_symbol=option_fields["option_symbol"],  # type: ignore[arg-type]
            underlying_symbol=option_fields["underlying_symbol"],  # type: ignore[arg-type]
            strike=option_fields["strike"],  # type: ignore[arg-type]
            expiration=option_fields["expiration"],  # type: ignore[arg-type]
            put_call=option_fields["put_call"],  # type: ignore[arg-type]
            contract_label=option_fields["contract_label"],  # type: ignore[arg-type]
            average_fill_price=avg_fill,
            premium_per_contract=premium_per_contract,
            total_cash=total_cash if total_cash is not None else total_premium,
            position_effect=leg.positionEffect,
        )

    def _activity_group_for_order(
        self,
        order: SchwabOrder,
        roll_groups: dict[int, ActivityGroupInfo],
    ) -> Optional[ActivityGroupInfo]:
        order_id = getattr(order, "orderId", None)
        if order_id is not None and order_id in roll_groups:
            return roll_groups[order_id]
        return spread_group_for_order(order)

    def to_recent_order_entry(
        self,
        order: SchwabOrder,
        *,
        roll_groups: dict[int, ActivityGroupInfo] | None = None,
    ) -> RecentOrderEntry:
        roll_groups = roll_groups or {}
        legs = order.orderLegCollection or []
        leg = order_primary_leg(order)
        instrument = leg.instrument if leg else None
        symbol = (
            order_underlying_symbol(leg)
            if leg
            else (instrument.symbol.upper() if instrument and instrument.symbol else "UNKNOWN")
        )

        side = (leg.instruction if leg and leg.instruction else "UNKNOWN").upper()
        qty = leg.quantity if leg and leg.quantity is not None else order.filledQuantity
        avg_fill = order_average_fill_price(order)
        premium_per_contract, total_premium = order_premium_fields(
            leg,
            fill_price_per_share=avg_fill,
            quantity=qty,
        )
        leg_count = len(legs) if legs else 1
        net_total_cash = order_net_total_cash(order) if leg_count > 1 else None
        total_cash = net_total_cash if net_total_cash is not None else order_total_cash(
            leg,
            fill_price_per_share=avg_fill,
            quantity=qty,
        )

        strategy_label = spread_group_for_order(order)
        strategy_label_text = strategy_label.label if strategy_label else None
        contract_label = leg_contract_label(leg) if leg else None

        activity_group = self._activity_group_for_order(order, roll_groups)

        return RecentOrderEntry(
            order_id=getattr(order, "orderId", None),
            symbol=symbol,
            fill_time=order_fill_time(order),
            side=side,
            quantity=qty,
            average_fill_price=avg_fill,
            order_type=order.orderType,
            position_effect=leg.positionEffect if leg else None,
            tax_lot_method=order.taxLotMethod,
            asset_type=order_asset_type(leg),
            description=instrument.description if instrument else None,
            premium_per_contract=premium_per_contract,
            total_premium=total_premium,
            total_cash=total_cash,
            leg_count=leg_count,
            strategy_label=strategy_label_text,
            contract_label=contract_label,
            strike=leg_strike(leg) if leg else None,
            expiration=leg_expiration(leg) if leg else None,
            put_call=leg_put_call(leg) if leg else None,
            legs=[self.to_recent_order_leg_entry(order, item) for item in legs],
            activity_group_id=activity_group.group_id if activity_group else None,
            activity_group_kind=activity_group.kind if activity_group else None,
            activity_group_label=activity_group.label if activity_group else None,
        )

    def to_recent_order_entries(
        self, orders: List[SchwabOrder]
    ) -> List[RecentOrderEntry]:
        roll_groups = detect_roll_groups(orders)
        return [
            self.to_recent_order_entry(order, roll_groups=roll_groups)
            for order in orders
        ]

    @staticmethod
    def suggest_analysis_actions(
        orders: List[SchwabOrder],
        *,
        symbol: Optional[str] = None,
        within_days: int = RECENT_ACTIVITY_DAYS,
    ) -> List[SuggestedAnalysisAction]:
        scoped = orders
        if symbol:
            scoped = [
                order
                for order in orders
                if order_relates_to_symbol(order, symbol.upper())
            ]

        recent = [
            order
            for order in scoped
            if is_order_within_days(order, within_days=within_days)
        ]
        if not recent:
            return []

        scope_label = symbol.upper() if symbol else "your portfolio"
        suggestions: List[SuggestedAnalysisAction] = []
        seen: set[AnalysisAction] = set()

        def add(action: AnalysisAction, reason: str, priority: int) -> None:
            if action in seen:
                return
            seen.add(action)
            suggestions.append(
                SuggestedAnalysisAction(
                    action=action,
                    label=action.label,
                    reason=reason,
                    priority=priority,
                )
            )

        has_sell = False
        has_buy = False
        for order in recent:
            leg = order_primary_leg(order)
            if not leg or not leg.instruction:
                continue
            side = leg.instruction.upper()
            if side in TAX_RELEVANT_SELL_INSTRUCTIONS:
                has_sell = True
            elif side in RISK_RELEVANT_BUY_INSTRUCTIONS:
                has_buy = True

        wash_flags = detect_wash_sale_flags(
            recent,
            symbol=symbol,
            window_days=WASH_SALE_WINDOW_DAYS,
        )
        if wash_flags:
            if symbol:
                add(
                    AnalysisAction.TAX_ANGLE,
                    (
                        f"Possible wash sale on {symbol.upper()}: you sold and bought "
                        f"within {WASH_SALE_WINDOW_DAYS} days — review tax lots and disallowed losses."
                    ),
                    priority=1,
                )
            else:
                flagged_symbols = sorted({flag.symbol for flag in wash_flags})
                preview = ", ".join(flagged_symbols[:3])
                suffix = "…" if len(flagged_symbols) > 3 else ""
                add(
                    AnalysisAction.TAX_ANGLE,
                    (
                        f"Possible wash sale on {preview}{suffix}: recent sell+buy pairs "
                        f"within {WASH_SALE_WINDOW_DAYS} days — review tax lots."
                    ),
                    priority=1,
                )
        elif has_sell:
            add(
                AnalysisAction.TAX_ANGLE,
                f"You sold {scope_label} recently — review tax implications and wash-sale rules.",
                priority=1,
            )

        last_fill = last_fill_time_for_symbol(recent, symbol=symbol)
        if last_fill:
            fill_label = last_fill.astimezone(timezone.utc).strftime("%b %d")
            add(
                AnalysisAction.WHAT_CHANGED,
                f"See what changed since your last {scope_label} fill on {fill_label}.",
                priority=2,
            )
        else:
            add(
                AnalysisAction.WHAT_CHANGED,
                f"Recent trade activity for {scope_label} — see what changed since your last fill.",
                priority=2,
            )

        if has_buy:
            add(
                AnalysisAction.RISK_CHECK,
                f"You added exposure in {scope_label} recently — check position size and downside risk.",
                priority=3,
            )

        return sorted(suggestions, key=lambda item: item.priority)

    @staticmethod
    def _sort_orders_newest_first(orders: List[SchwabOrder]) -> List[SchwabOrder]:
        return sorted(
            orders,
            key=lambda order: order_fill_time(order) or datetime.min,
            reverse=True,
        )

    def build_recent_activity_summary(
        self,
        account_number: str,
        access_token: str,
        *,
        user_id: Optional[str] = None,
        days_back: int = DEFAULT_DAYS_BACK,
        refresh: bool = False,
    ) -> RecentActivitySummary:
        orders = self._fetch_filled_orders(
            account_number=account_number,
            access_token=access_token,
            user_id=user_id,
            days_back=days_back,
            refresh=refresh,
        )
        sorted_orders = self._sort_orders_newest_first(orders)
        roll_groups = detect_roll_groups(sorted_orders)

        symbol_stats: dict[str, dict[str, object]] = {}
        for order in orders:
            fill_time = order_fill_time(order)
            for sym in order_symbols(order):
                stats = symbol_stats.setdefault(
                    sym,
                    {"count": 0, "last_fill_time": None},
                )
                stats["count"] = int(stats["count"]) + 1
                last_fill = stats["last_fill_time"]
                if fill_time and (last_fill is None or fill_time > last_fill):
                    stats["last_fill_time"] = fill_time

        symbols_traded = sorted(
            [
                RecentActivitySymbolSummary(
                    symbol=sym,
                    order_count=int(stats["count"]),
                    last_fill_time=stats["last_fill_time"],  # type: ignore[arg-type]
                )
                for sym, stats in symbol_stats.items()
            ],
            key=lambda item: item.last_fill_time or datetime.min,
            reverse=True,
        )[:PORTFOLIO_SYMBOLS_LIMIT]

        recent_order_count = sum(
            1
            for order in orders
            if is_order_within_days(order, within_days=RECENT_ACTIVITY_DAYS)
        )

        return RecentActivitySummary(
            days_back=days_back,
            total_orders=len(orders),
            recent_order_count=recent_order_count,
            symbols_traded=symbols_traded,
            latest_orders=[
                self.to_recent_order_entry(order, roll_groups=roll_groups)
                for order in sorted_orders[:PORTFOLIO_LATEST_ORDERS_LIMIT]
            ],
            suggested_actions=self.suggest_analysis_actions(
                orders,
                within_days=RECENT_ACTIVITY_DAYS,
            ),
        )

    def build_recent_orders_response(
        self,
        account_number: str,
        access_token: str,
        *,
        user_id: Optional[str] = None,
        symbol: Optional[str] = None,
        days_back: int = DEFAULT_DAYS_BACK,
        refresh: bool = False,
    ) -> RecentOrdersResponse:
        orders = self.get_filled_orders(
            account_number=account_number,
            access_token=access_token,
            user_id=user_id,
            symbol=symbol,
            days_back=days_back,
            refresh=refresh,
        )

        sorted_orders = self._sort_orders_newest_first(orders)
        roll_groups = detect_roll_groups(sorted_orders)
        entries = [
            self.to_recent_order_entry(order, roll_groups=roll_groups)
            for order in sorted_orders
        ]

        activity_by_symbol: dict[str, int] = {}
        for order in orders:
            for sym in order_symbols(order):
                activity_by_symbol[sym] = activity_by_symbol.get(sym, 0) + 1

        return RecentOrdersResponse(
            days_back=days_back,
            symbol=symbol.upper() if symbol else None,
            orders=entries,
            suggested_actions=self.suggest_analysis_actions(
                orders,
                symbol=symbol,
                within_days=RECENT_ACTIVITY_DAYS,
            ),
            activity_by_symbol=activity_by_symbol,
        )
