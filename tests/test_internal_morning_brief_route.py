import os
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.morning_brief_delivery_service import (
    MorningBriefDispatchResult,
    MorningBriefPrewarmResult,
)


def test_prewarm_morning_briefs_requires_cron_secret(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")
    client = TestClient(app)

    response = client.post("/api/v1/internal/prewarm-morning-briefs")
    assert response.status_code == 401


def test_prewarm_morning_briefs_returns_counts(monkeypatch):
    monkeypatch.setenv("CRON_SECRET", "test-secret")

    delivery_service = MagicMock()
    delivery_service.prewarm_all.return_value = MorningBriefPrewarmResult(
        attempted=2,
        warmed=1,
        skipped=1,
        failed=0,
        errors=[],
    )
    app.state.morning_brief_delivery_service = delivery_service

    client = TestClient(app)
    response = client.post(
        "/api/v1/internal/prewarm-morning-briefs",
        headers={"X-Cron-Secret": "test-secret"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "attempted": 2,
        "warmed": 1,
        "skipped": 1,
        "failed": 0,
        "errors": [],
    }
