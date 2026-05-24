from datetime import datetime
from typing import List, Optional

from app.adapters.cache.recent_orders_cache import RecentOrdersCache
from app.broker.order_utils import (
    is_order_within_days,
    order_asset_type,
    order_average_fill_price,
    order_fill_time,
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
    RecentOrdersResponse,
    SuggestedAnalysisAction,
)
from app.models.schwab_order_models import SchwabOrder

DEFAULT_DAYS_BACK = 30
RECENT_ACTIVITY_DAYS = 7
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

    def to_recent_order_entry(self, order: SchwabOrder) -> RecentOrderEntry:
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
        total_cash = order_total_cash(
            leg,
            fill_price_per_share=avg_fill,
            quantity=qty,
        )

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
        )

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

        if has_sell:
            add(
                AnalysisAction.TAX_ANGLE,
                f"You sold {scope_label} recently — review tax implications and wash-sale rules.",
                priority=1,
            )

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
                self.to_recent_order_entry(order)
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
        entries = [self.to_recent_order_entry(order) for order in sorted_orders]

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
