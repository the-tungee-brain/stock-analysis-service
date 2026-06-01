from unittest.mock import MagicMock

from app.services.account_deletion_service import AccountDeletionService


def _build_service() -> tuple[AccountDeletionService, dict[str, MagicMock]]:
    deps = {
        "schwab_auth_service": MagicMock(),
        "chat_sessions_builder": MagicMock(),
        "chat_messages_builder": MagicMock(),
        "app_user_adapter": MagicMock(),
        "user_investment_profile_adapter": MagicMock(),
        "user_strategy_journey_adapter": MagicMock(),
        "watchlist_adapter": MagicMock(),
        "alert_history_adapter": MagicMock(),
        "portfolio_snapshot_adapter": MagicMock(),
        "morning_brief_delivery_adapter": MagicMock(),
        "waitlist_adapter": MagicMock(),
        "recent_orders_cache": MagicMock(),
        "portfolio_brief_cache": MagicMock(),
    }
    service = AccountDeletionService(**deps)
    return service, deps


def test_delete_account_cleans_up_all_user_data():
    service, deps = _build_service()
    deps["chat_sessions_builder"].get_sessions_by_user_id.return_value = []

    service.delete_account("user-123")

    deps["schwab_auth_service"].disconnect_user.assert_called_once_with(user_id="user-123")
    deps["user_investment_profile_adapter"].delete_by_user_id.assert_called_once_with(
        "user-123"
    )
    deps["user_strategy_journey_adapter"].delete_by_user_id.assert_called_once_with(
        "user-123"
    )
    deps["watchlist_adapter"].delete_by_user_id.assert_called_once_with("user-123")
    deps["alert_history_adapter"].delete_by_user_id.assert_called_once_with("user-123")
    deps["portfolio_snapshot_adapter"].delete_by_user_id.assert_called_once_with(
        "user-123"
    )
    deps["morning_brief_delivery_adapter"].delete_by_user_id.assert_called_once_with(
        "user-123"
    )
    deps["waitlist_adapter"].delete_by_identity_sub.assert_called_once_with("user-123")
    deps["recent_orders_cache"].invalidate_user.assert_called_once_with(
        user_id="user-123"
    )
    deps["portfolio_brief_cache"].invalidate_user.assert_called_once_with(
        user_id="user-123"
    )
    deps["app_user_adapter"].delete_by_identity_sub.assert_called_once_with("user-123")


def test_delete_account_removes_chat_sessions_in_batches():
    service, deps = _build_service()
    session_one = MagicMock(id="session-1")
    session_two = MagicMock(id="session-2")
    deps["chat_sessions_builder"].get_sessions_by_user_id.side_effect = [
        [session_one],
        [session_two],
        [],
    ]

    service.delete_account("user-456")

    deps["chat_messages_builder"].delete_messages_by_session.assert_any_call("session-1")
    deps["chat_messages_builder"].delete_messages_by_session.assert_any_call("session-2")
    deps["chat_sessions_builder"].delete_session.assert_any_call("session-1")
    deps["chat_sessions_builder"].delete_session.assert_any_call("session-2")
    assert deps["chat_sessions_builder"].get_sessions_by_user_id.call_count == 3
