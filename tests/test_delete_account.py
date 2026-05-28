from unittest.mock import MagicMock

from app.api.delete_account_route import delete_account
from app.services.account_deletion_service import AccountDeletionService


def test_delete_account_route():
    service = MagicMock(spec=AccountDeletionService)

    result = delete_account(
        user_id="user-123",
        account_deletion_service=service,
    )

    service.delete_account.assert_called_once_with("user-123")
    assert result == {"deleted": True}
