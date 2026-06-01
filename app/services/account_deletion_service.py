from __future__ import annotations

from app.adapters.cache.portfolio_brief_cache import PortfolioBriefCache
from app.adapters.cache.recent_orders_cache import RecentOrdersCache
from app.adapters.portfolio.alert_history_adapter import AlertHistoryAdapter
from app.adapters.portfolio.morning_brief_delivery_adapter import (
    MorningBriefDeliveryAdapter,
)
from app.adapters.portfolio.portfolio_snapshot_adapter import PortfolioSnapshotAdapter
from app.adapters.user.app_user_adapter import AppUserAdapter
from app.adapters.user.user_investment_profile_adapter import (
    UserInvestmentProfileAdapter,
)
from app.adapters.user.user_strategy_journey_adapter import UserStrategyJourneyAdapter
from app.adapters.user.watchlist_adapter import WatchlistAdapter
from app.adapters.user.waitlist_adapter import WaitlistAdapter
from app.builders.chat_messages_builder import ChatMessagesBuilder
from app.builders.chat_sessions_builder import ChatSessionsBuilder
from app.services.schwab_auth_service import SchwabAuthService


class AccountDeletionService:
    def __init__(
        self,
        *,
        schwab_auth_service: SchwabAuthService,
        chat_sessions_builder: ChatSessionsBuilder,
        chat_messages_builder: ChatMessagesBuilder,
        app_user_adapter: AppUserAdapter,
        user_investment_profile_adapter: UserInvestmentProfileAdapter,
        user_strategy_journey_adapter: UserStrategyJourneyAdapter,
        watchlist_adapter: WatchlistAdapter,
        alert_history_adapter: AlertHistoryAdapter,
        portfolio_snapshot_adapter: PortfolioSnapshotAdapter,
        morning_brief_delivery_adapter: MorningBriefDeliveryAdapter,
        waitlist_adapter: WaitlistAdapter,
        recent_orders_cache: RecentOrdersCache,
        portfolio_brief_cache: PortfolioBriefCache,
    ):
        self.schwab_auth_service = schwab_auth_service
        self.chat_sessions_builder = chat_sessions_builder
        self.chat_messages_builder = chat_messages_builder
        self.app_user_adapter = app_user_adapter
        self.user_investment_profile_adapter = user_investment_profile_adapter
        self.user_strategy_journey_adapter = user_strategy_journey_adapter
        self.watchlist_adapter = watchlist_adapter
        self.alert_history_adapter = alert_history_adapter
        self.portfolio_snapshot_adapter = portfolio_snapshot_adapter
        self.morning_brief_delivery_adapter = morning_brief_delivery_adapter
        self.waitlist_adapter = waitlist_adapter
        self.recent_orders_cache = recent_orders_cache
        self.portfolio_brief_cache = portfolio_brief_cache

    def delete_account(self, user_id: str) -> None:
        self.schwab_auth_service.disconnect_user(user_id=user_id)
        self._delete_chat_data(user_id)
        self.user_investment_profile_adapter.delete_by_user_id(user_id)
        self.user_strategy_journey_adapter.delete_by_user_id(user_id)
        self.watchlist_adapter.delete_by_user_id(user_id)
        self.alert_history_adapter.delete_by_user_id(user_id)
        self.portfolio_snapshot_adapter.delete_by_user_id(user_id)
        self.morning_brief_delivery_adapter.delete_by_user_id(user_id)
        self.waitlist_adapter.delete_by_identity_sub(user_id)
        self.recent_orders_cache.invalidate_user(user_id=user_id)
        self.portfolio_brief_cache.invalidate_user(user_id=user_id)
        self.app_user_adapter.delete_by_identity_sub(user_id)

    def _delete_chat_data(self, user_id: str) -> None:
        while True:
            sessions = self.chat_sessions_builder.get_sessions_by_user_id(
                user_id=user_id,
                limit=100,
            )
            if not sessions:
                break

            for session in sessions:
                if session.id is None:
                    continue
                self.chat_messages_builder.delete_messages_by_session(session.id)
                self.chat_sessions_builder.delete_session(session.id)
