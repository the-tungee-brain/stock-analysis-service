from __future__ import annotations

from pathlib import Path

import logging
import pytest

from app.api.get_emerging_leaders_route import get_emerging_leaders
from app.services.emerging_leaders_service import (
    SNAPSHOT_UNAVAILABLE_DETAIL,
    build_emerging_leaders,
)
from app.storage.emerging_leaders_store import EmergingLeadersStore


def _result(symbol: str, rank: int) -> dict:
    return {
        "rank": rank,
        "symbol": symbol,
        "setup_quality_score": 75,
        "setup_stage": "TIGHTENING",
        "setup_stage_label": "Stage 2: Tightening",
        "compression_velocity": 70,
        "compression_velocity_label": "High",
        "why_it_ranks": f"{symbol} is tightening.",
        "positive_factors": ["Volume contraction"],
        "missing_factors": ["Breakout trigger"],
        "next_confirmation": ["Close above resistance"],
        "components": {"compression_velocity": 70},
    }


@pytest.fixture
def snapshot_store(tmp_path: Path) -> EmergingLeadersStore:
    return EmergingLeadersStore(tmp_path / "ranking.db")


@pytest.fixture(autouse=True)
def _default_precomputed_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMERGING_LEADERS_SERVING_MODE", "precomputed")


def _completed_run(
    store: EmergingLeadersStore,
    *,
    run_id: str,
    generated_at: str,
    symbols: list[str],
) -> None:
    store.start_run(run_id=run_id, generated_at=generated_at)
    store.complete_run(
        run_id=run_id,
        as_of_date="2026-06-05",
        generated_at=generated_at,
        universe_snapshot_id="snap-1",
        ranking_run_id="ranking-run-1",
        symbols_with_data=200,
        candidates_scanned=188,
        excluded_top_movers=12,
        evaluations_computed=len(symbols),
        duration_ms=1234,
        results=[_result(symbol, idx) for idx, symbol in enumerate(symbols, start=1)],
    )


def test_latest_completed_snapshot_is_used(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    _completed_run(
        snapshot_store,
        run_id="old",
        generated_at="2026-06-05T10:00:00+00:00",
        symbols=["OLD"],
    )
    _completed_run(
        snapshot_store,
        run_id="new",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=["NEW"],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    response = build_emerging_leaders(limit=20)

    assert response.timestamp == "2026-06-05T12:00:00+00:00"
    assert [item.symbol for item in response.items] == ["NEW"]


def test_failed_and_running_snapshots_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    snapshot_store.start_run(
        run_id="running",
        generated_at="2026-06-05T13:00:00+00:00",
    )
    snapshot_store.start_run(
        run_id="failed",
        generated_at="2026-06-05T12:30:00+00:00",
    )
    snapshot_store.fail_run(
        run_id="failed",
        error_message="boom",
        duration_ms=20,
    )
    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=["AAA"],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    response = build_emerging_leaders(limit=20)

    assert response.timestamp == "2026-06-05T12:00:00+00:00"
    assert [item.symbol for item in response.items] == ["AAA"]


def test_response_json_shape_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=["AAA"],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    payload = build_emerging_leaders(limit=20).model_dump(mode="json", by_alias=True)

    assert set(payload) == {
        "asOfDate",
        "timestamp",
        "universeScanned",
        "symbolsWithData",
        "evaluationsComputed",
        "excludedTopMovers",
        "items",
    }
    assert set(payload["items"][0]) == {
        "rank",
        "symbol",
        "setupQualityScore",
        "setupStage",
        "setupStageLabel",
        "compressionVelocity",
        "compressionVelocityLabel",
        "whyItRanks",
        "positiveFactors",
        "missingFactors",
        "nextConfirmation",
    }


def test_limit_truncates_stored_top_100(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=[f"S{i:03d}" for i in range(100)],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    response = build_emerging_leaders(limit=3)

    assert [item.symbol for item in response.items] == ["S000", "S001", "S002"]


@pytest.mark.asyncio
async def test_no_snapshot_in_precomputed_mode_returns_503(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    with pytest.raises(Exception) as exc:
        await get_emerging_leaders(limit=20, user_id="user-1")

    assert getattr(exc.value, "status_code") == 503
    assert getattr(exc.value, "detail") == SNAPSHOT_UNAVAILABLE_DETAIL


def test_live_emergency_calls_live_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMERGING_LEADERS_SERVING_MODE", "live_emergency")
    called = {"live": False}

    def fake_live(*, limit: int):
        called["live"] = True
        raise LookupError("live called")

    monkeypatch.setattr(
        "app.services.emerging_leaders_service.build_emerging_leaders_live",
        fake_live,
    )

    with pytest.raises(LookupError):
        build_emerging_leaders(limit=20)

    assert called["live"] is True


def test_precomputed_with_live_fallback_only_scans_when_no_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.setenv(
        "EMERGING_LEADERS_SERVING_MODE",
        "precomputed_with_live_fallback",
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )
    called = {"live": 0}

    def fake_live(*, limit: int):
        called["live"] += 1
        raise LookupError("live fallback called")

    monkeypatch.setattr(
        "app.services.emerging_leaders_service.build_emerging_leaders_live",
        fake_live,
    )

    with pytest.raises(LookupError):
        build_emerging_leaders(limit=20)

    assert called["live"] == 1

    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=["AAA"],
    )
    response = build_emerging_leaders(limit=20)

    assert [item.symbol for item in response.items] == ["AAA"]
    assert called["live"] == 1


def test_live_scoring_is_not_called_when_snapshot_exists(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T12:00:00+00:00",
        symbols=["AAA"],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    def fail_live(*, limit: int):
        raise AssertionError("live scoring should not be called")

    monkeypatch.setattr(
        "app.services.emerging_leaders_service.build_emerging_leaders_live",
        fail_live,
    )

    response = build_emerging_leaders(limit=20)

    assert [item.symbol for item in response.items] == ["AAA"]


def test_stale_snapshot_is_served_with_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.setenv("EMERGING_LEADERS_MAX_SNAPSHOT_AGE_HOURS", "1")
    _completed_run(
        snapshot_store,
        run_id="completed",
        generated_at="2020-01-01T12:00:00+00:00",
        symbols=["AAA"],
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_service.open_emerging_leaders_store",
        lambda: snapshot_store,
    )

    with caplog.at_level(logging.WARNING):
        response = build_emerging_leaders(limit=20)

    assert [item.symbol for item in response.items] == ["AAA"]
    assert "Emerging Leaders snapshot is stale" in caplog.text
