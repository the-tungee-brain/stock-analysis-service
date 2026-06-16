from unittest.mock import MagicMock

import pytest
import requests
from fastapi import HTTPException

from app.api.get_recent_orders_route import get_recent_orders


def test_recent_orders_schwab_timeout_returns_retryable_504():
    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.return_value = {
        "account": MagicMock(
            securitiesAccount=MagicMock(accountNumber="12345678"),
        )
    }

    schwab_auth_service = MagicMock()
    schwab_auth_service.get_valid_token_by_user_id.return_value = MagicMock(
        access_token="token"
    )

    transaction_service = MagicMock()
    transaction_service.build_recent_orders_response.side_effect = (
        requests.exceptions.ReadTimeout("Schwab orders request timed out")
    )

    with pytest.raises(HTTPException) as exc_info:
        get_recent_orders(
            user_id="user-1",
            transaction_service=transaction_service,
            portfolio_service=portfolio_service,
            schwab_auth_service=schwab_auth_service,
        )

    assert exc_info.value.status_code == 504
    assert exc_info.value.detail == {
        "message": "Schwab is temporarily unavailable. Please retry shortly.",
        "retryable": True,
    }
