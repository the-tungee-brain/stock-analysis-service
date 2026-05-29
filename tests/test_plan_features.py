import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.api.get_dividend_history_route import get_dividend_history
from app.api.wheel_backtest_route import get_wheel_backtest
from app.core.llm_config import settings
from app.core.plan_features import paid_features_for_user
from tests.test_dividend_history_route import _sample_context


def test_paid_features_for_free_user(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    features = paid_features_for_user("user-free")
    assert features["wheel_backtest"] is False
    assert features["dividend_snowball"] is False
    assert features["news_ai"] is False
    assert features["financial_strength"] is False
    assert features["earnings_ai"] is False


def test_paid_features_for_pro_user(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"user-pro"}))
    features = paid_features_for_user("user-pro")
    assert features["wheel_backtest"] is True
    assert features["dividend_snowball"] is True
    assert features["news_ai"] is True
    assert features["financial_strength"] is True
    assert features["earnings_ai"] is True


def test_wheel_backtest_requires_pro(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    service = MagicMock()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            get_wheel_backtest(
                user_id="user-free",
                symbol="AAPL",
                years=5,
                target_delta_min=0.2,
                target_delta_max=0.3,
                dte_days=30,
                contracts=1,
                maintain_one_lot=True,
                call_strike_mode="delta",
                service=service,
            )
        )

    assert exc.value.status_code == 403
    service.run_backtest.assert_not_called()


def test_free_dividend_history_omits_snowball(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    service = MagicMock()
    service.build_history_context.return_value = _sample_context()

    asyncio.run(
        get_dividend_history(
            user_id="user-free",
            symbol="SCHD",
            shares=100,
            reinvest_dividends=True,
            project_years=25,
            annual_contribution_usd=15000.0,
            dividend_research_service=service,
        )
    )

    call = service.build_history_context.call_args
    assert call.args[0] == "SCHD"
    assert call.kwargs["shares"] == 100
    assert call.kwargs["reinvest_dividends"] is False
    assert call.kwargs["project_years"] is None
    assert call.kwargs["annual_contribution_usd"] == 0.0
    assert call.kwargs["include_snowball"] is False
