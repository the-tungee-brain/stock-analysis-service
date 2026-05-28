from app.api.get_account_plan_route import get_account_plan
from app.core.llm_config import settings


def test_free_user_plan(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    monkeypatch.setattr(settings, "OPENAI_FREE_MODEL", "gpt-4.1-mini")

    result = get_account_plan(user_id="user-free")

    assert result["plan"] == "free"
    assert result["isPaid"] is False
    assert result["freeModel"] == "gpt-4.1-mini"


def test_paid_user_plan(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"user-paid"}))

    result = get_account_plan(user_id="user-paid")

    assert result["plan"] == "pro"
    assert result["isPaid"] is True
