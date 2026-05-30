from unittest.mock import MagicMock

from app.api.get_account_plan_route import get_account_plan
from app.core import paid_access
from app.core.llm_config import settings


def _user(*, sub: str, email: str = "user@example.com"):
    user = MagicMock()
    user.identity_sub = sub
    user.email = email
    return user


def test_free_user_plan(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    monkeypatch.setattr(settings, "PAID_USER_EMAILS", frozenset())
    monkeypatch.setattr(settings, "OPENAI_FREE_MODEL", "gpt-4.1-mini")
    paid_access._email_for_identity.cache_clear()

    result = get_account_plan(user=_user(sub="user-free"))

    assert result["plan"] == "free"
    assert result["isPaid"] is False
    assert result["identitySub"] == "user-free"
    assert result["freeModel"] == "gpt-4.1-mini"
    assert result["freeModels"] == ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-nano"]
    assert "gpt-4o" in result["proOnlyModels"]
    assert "gpt-5.1" in result["proOnlyModels"]
    assert result["allowedModels"] == result["freeModels"]
    assert len(result["chatModels"]) >= 8
    assert result["features"]["wheel_backtest"] is False
    assert result["features"]["dividend_snowball"] is False
    assert result["features"]["news_ai"] is False
    assert result["features"]["financial_strength"] is False
    assert result["features"]["earnings_ai"] is False


def test_paid_user_plan(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"user-paid"}))
    monkeypatch.setattr(settings, "PAID_USER_EMAILS", frozenset())
    paid_access._email_for_identity.cache_clear()

    result = get_account_plan(user=_user(sub="user-paid"))

    assert result["plan"] == "pro"
    assert result["isPaid"] is True
    assert "gpt-4o" in result["allowedModels"]
    assert result["allowedModels"] == result["paidModels"]
    assert result["features"]["wheel_backtest"] is True
    assert result["features"]["dividend_snowball"] is True
    assert result["features"]["news_ai"] is True
    assert result["features"]["financial_strength"] is True
    assert result["features"]["earnings_ai"] is True
