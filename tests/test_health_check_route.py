from fastapi.testclient import TestClient

from app.core.llm_config import settings
from app.main import app


def _restore_pattern_state(loaded_model, error):
    app.state.pattern_loaded_model = loaded_model
    app.state.pattern_model_error = error


def test_health_unhealthy_when_paid_pattern_model_missing(monkeypatch):
    original_loaded = getattr(app.state, "pattern_loaded_model", None)
    original_error = getattr(app.state, "pattern_model_error", None)
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"paid-user"}))
    monkeypatch.setattr(settings, "PAID_USER_EMAILS", frozenset())
    try:
        app.state.pattern_loaded_model = None
        app.state.pattern_model_error = "Model artifact not found"

        response = TestClient(app).get("/health")

        assert response.status_code == 503
        payload = response.json()
        assert payload["status"] == "unhealthy"
        assert payload["patternModel"]["loaded"] is False
        assert payload["patternModel"]["error"] == "Model artifact not found"
    finally:
        _restore_pattern_state(original_loaded, original_error)


def test_health_ok_when_no_paid_pattern_users_configured(monkeypatch):
    original_loaded = getattr(app.state, "pattern_loaded_model", None)
    original_error = getattr(app.state, "pattern_model_error", None)
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    monkeypatch.setattr(settings, "PAID_USER_EMAILS", frozenset())
    try:
        app.state.pattern_loaded_model = None
        app.state.pattern_model_error = "Model artifact not found"

        response = TestClient(app).get("/health")

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["patternModel"]["loaded"] is False
    finally:
        _restore_pattern_state(original_loaded, original_error)
