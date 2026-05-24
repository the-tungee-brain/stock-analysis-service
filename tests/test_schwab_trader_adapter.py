from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter


def _adapter_with_session() -> tuple[SchwabTraderAdapter, MagicMock]:
    session = MagicMock()
    adapter = SchwabTraderAdapter(session=session, base_uri="https://api.test/trader/v1")
    return adapter, session


def test_get_orders_uses_account_hash_not_account_number():
    adapter, session = _adapter_with_session()

    account_numbers_response = MagicMock()
    account_numbers_response.raise_for_status = MagicMock()
    account_numbers_response.json.return_value = [
        {"accountNumber": "25859996", "hashValue": "HASH-ABC123"}
    ]

    orders_response = MagicMock()
    orders_response.ok = True
    orders_response.json.return_value = []

    session.get.side_effect = [account_numbers_response, orders_response]

    orders = adapter.get_orders(
        account_number="25859996",
        access_token="token",
        status="FILLED",
        days_back=30,
    )

    assert orders == []
    assert session.get.call_count == 2

    numbers_call = session.get.call_args_list[0]
    assert numbers_call.args[0] == "https://api.test/trader/v1/accounts/accountNumbers"

    orders_call = session.get.call_args_list[1]
    assert orders_call.args[0] == "https://api.test/trader/v1/accounts/HASH-ABC123/orders"
    assert orders_call.kwargs["params"]["status"] == "FILLED"


def test_resolve_account_hash_is_cached():
    adapter, session = _adapter_with_session()

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [
        {"accountNumber": "25859996", "hashValue": "HASH-ABC123"}
    ]
    session.get.return_value = response

    first = adapter.resolve_account_hash("token", "25859996")
    second = adapter.resolve_account_hash("token", "25859996")

    assert first == second == "HASH-ABC123"
    assert session.get.call_count == 1


def test_get_orders_raises_when_account_hash_missing():
    adapter, session = _adapter_with_session()

    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = [
        {"accountNumber": "99999999", "hashValue": "OTHER-HASH"}
    ]
    session.get.return_value = response

    with pytest.raises(ValueError, match="No Schwab account hash found"):
        adapter.get_orders(
            account_number="25859996",
            access_token="token",
            days_back=7,
        )


def test_get_orders_clamps_days_back_to_schwab_limit():
    adapter, session = _adapter_with_session()

    account_numbers_response = MagicMock()
    account_numbers_response.raise_for_status = MagicMock()
    account_numbers_response.json.return_value = [
        {"accountNumber": "123", "hashValue": "HASH-123"}
    ]

    orders_response = MagicMock()
    orders_response.ok = True
    orders_response.json.return_value = []

    session.get.side_effect = [account_numbers_response, orders_response]

    adapter.get_orders(
        account_number="123",
        access_token="token",
        days_back=365,
    )

    orders_call = session.get.call_args_list[1]
    from_time = orders_call.kwargs["params"]["fromEnteredTime"]
    to_time = orders_call.kwargs["params"]["toEnteredTime"]

    start = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
    end = datetime.fromisoformat(to_time.replace("Z", "+00:00"))
    assert (end - start).days <= 60
