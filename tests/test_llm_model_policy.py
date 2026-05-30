from app.core.llm_config import settings
from app.core.llm_model_policy import (
    chat_model_policy_for_client,
    is_paid_user,
    resolve_llm_model,
)


def test_free_user_advanced_model_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    assert resolve_llm_model("gpt-5.4", "user-free") == settings.OPENAI_FREE_MODEL


def test_free_user_can_pick_standard_model(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    assert resolve_llm_model("gpt-4.1-mini", "user-free") == "gpt-4.1-mini"


def test_free_user_pro_standard_models_fall_back(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    assert resolve_llm_model("gpt-4o", "user-free") == settings.OPENAI_FREE_MODEL
    assert resolve_llm_model("gpt-5.1", "user-free") == settings.OPENAI_FREE_MODEL


def test_chat_model_policy_for_client_free():
    policy = chat_model_policy_for_client(is_paid=False)
    assert "gpt-4o" in policy["proOnlyModels"]
    assert "gpt-4o" not in policy["freeModels"]
    assert policy["allowedModels"] == policy["freeModels"]


def test_chat_model_policy_for_client_paid():
    policy = chat_model_policy_for_client(is_paid=True)
    assert "gpt-4o" in policy["allowedModels"]
    assert len(policy["chatModels"]) >= 8


def test_free_user_can_pick_simple_model(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    assert resolve_llm_model("gpt-5-nano", "user-free") == "gpt-5-nano"


def test_paid_user_can_request_allowed_model(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"user-paid"}))
    assert resolve_llm_model("gpt-5-mini", "user-paid") == "gpt-5-mini"


def test_paid_user_invalid_model_falls_back(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"user-paid"}))
    assert resolve_llm_model("gpt-unknown", "user-paid") == settings.OPENAI_MODEL


def test_is_paid_user():
    original = settings.PAID_USER_IDS
    try:
        settings.PAID_USER_IDS = frozenset({"abc"})
        assert is_paid_user("abc") is True
        assert is_paid_user("xyz") is False
    finally:
        settings.PAID_USER_IDS = original
