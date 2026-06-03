"""Product API v1 contract and health reader tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.api.product.models import API_VERSION, RankingsTopResponseV1, SystemHealthResponseV1
from app.services.pipeline_status_reader import PipelineStatusReader


def test_api_version_constant():
    assert API_VERSION == "v1"


def test_rankings_response_schema():
    payload = RankingsTopResponseV1(
        timestamp="2026-06-01T00:00:00+00:00",
        run_id="run-1",
        as_of_date="2026-06-01",
        regime_id="risk_on_trend",
        items=[],
    )
    assert payload.api_version == "v1"
    assert payload.regime_id == "risk_on_trend"


def test_health_reader_missing_db(tmp_path: Path):
    reader = PipelineStatusReader(tmp_path / "missing.db")
    snap = reader.get_status()
    assert snap.system_status == "failing"


def test_health_reader_with_ranking_run(tmp_path: Path):
    db = tmp_path / "rank.db"
    import sqlite3

    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE ranking_runs (
          run_id TEXT PRIMARY KEY, as_of_date TEXT, model_backend TEXT,
          universe_snapshot_id TEXT, symbol_count INTEGER, created_at TEXT, regime_id TEXT
        );
        CREATE TABLE universe_snapshots (
          snapshot_id TEXT PRIMARY KEY, created_at TEXT, symbol_count INTEGER
        );
        CREATE TABLE portfolio_snapshots (
          portfolio_id TEXT PRIMARY KEY, ranking_run_id TEXT, as_of_date TEXT,
          sizing_mode TEXT, created_at TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO ranking_runs VALUES (?,?,?,?,?,?,?)",
        ("r1", "2026-06-01", "composite", "u1", 100, "2026-06-02T10:00:00+00:00", "risk_on_chop"),
    )
    conn.execute(
        "INSERT INTO universe_snapshots VALUES (?,?,?)",
        ("u1", "2026-06-01", 500),
    )
    conn.commit()
    conn.close()

    snap = PipelineStatusReader(db).get_status()
    assert snap.system_status == "ok"
    assert snap.universe_size == 500
    assert snap.regime_id == "risk_on_chop"
