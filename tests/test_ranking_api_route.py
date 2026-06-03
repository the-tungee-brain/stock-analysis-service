"""API route tests for precomputed rankings."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from ranking_pipeline.storage.sqlite import RankingStore


@pytest.fixture
def ranking_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db = tmp_path / "rank.db"
    monkeypatch.setenv("RANKING_DB_PATH", str(db))
    store = RankingStore(db)
    store.save_universe_snapshot(
        "snap-1",
        [{"symbol": "AAPL", "passed_filters": True, "last_close": 1.0}],
    )
    store.save_ranking_run(
        "run-test",
        "2026-06-01",
        "composite",
        "snap-1",
        [
            {
                "symbol": "AAPL",
                "rank": 1,
                "composite_score": 1.0,
                "ml_probability": 0.65,
                "expected_excess_return": 0.01,
                "final_score": 0.9,
                "contributions": {"relative_strength": 0.4},
            }
        ],
    )
    return db


def test_rankings_top_route_requires_auth():
    client = TestClient(app)
    resp = client.get("/api/v1/rankings/top")
    assert resp.status_code in (401, 403)
