from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.builders.emerging_leaders_engine import (
    EmergingLeaderEvaluation,
    SetupComponentScores,
)
from app.services.emerging_leaders_precompute_service import (
    precompute_emerging_leaders_snapshot,
)
from app.services.emerging_leaders_service import build_emerging_leaders_live
from app.storage.emerging_leaders_store import EmergingLeadersStore
from ranking_pipeline.storage.sqlite import UniverseMemberRecord


def _member(symbol: str) -> UniverseMemberRecord:
    return UniverseMemberRecord(
        symbol=symbol,
        market_cap=1_000_000_000,
        avg_dollar_volume_20d=1_000_000,
        ranking_score=None,
    )


def _components() -> SetupComponentScores:
    return SetupComponentScores(
        volatility_contraction=70,
        range_tightening=70,
        resistance_tests_score=70,
        resistance_tests=3,
        volume_dryup=70,
        rs_improvement=50,
        accumulation=50,
        base_quality=70,
        breakout_proximity=70,
        ret_5d=0.01,
        ret_10d=0.02,
        ret_20d=0.03,
        distance_from_breakout_pct=4.0,
        resistance_level=100,
        tightening_trend=70,
        vol_contraction_trend=70,
        vol_contraction_deep=70,
        consolidation_structure=70,
        volume_dryup_trend=70,
        base_eligibility_signals=3,
        dormancy_days=20,
        base_age=30,
        failed_resistance_tests=2,
        rsi_14=55,
        setup_purity_score=70,
        compression_velocity=70,
        momentum_leader_like=False,
    )


def _evaluation(symbol: str) -> EmergingLeaderEvaluation:
    return EmergingLeaderEvaluation(
        symbol=symbol,
        setup_quality_score=75,
        setup_stage="TIGHTENING",
        why_it_ranks=f"{symbol} is tightening.",
        positive_factors=["Volume contraction"],
        missing_factors=["Breakout trigger"],
        next_confirmation=["Close above resistance"],
        sort_priority=120,
        components=_components(),
    )


class _FakeRankingStore:
    def __init__(
        self,
        symbols: list[str],
        *,
        top_movers: list[str] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.symbols = symbols
        self.top_movers = top_movers or []
        self.run_id = run_id

    def active_snapshot_id(self) -> str:
        return "snap-1"

    def load_passed_universe_members(
        self,
        snapshot_id: str | None = None,
    ) -> list[UniverseMemberRecord]:
        assert snapshot_id == "snap-1"
        return [_member(symbol) for symbol in self.symbols]

    def latest_run_id(self) -> str | None:
        return self.run_id

    def get_run_meta(self, run_id: str) -> dict[str, Any]:
        assert run_id == self.run_id
        return {"as_of_date": "2026-06-05"}

    def get_ranking_results(
        self,
        run_id: str,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        assert run_id == self.run_id
        return [{"symbol": symbol} for symbol in self.top_movers[:limit]]

    def get_latest_ranking_run(self) -> None:
        return None

    def count_ranking_results(self, run_id: str) -> int:
        return 0


@pytest.fixture
def snapshot_store(tmp_path: Path) -> EmergingLeadersStore:
    return EmergingLeadersStore(tmp_path / "ranking.db")


@pytest.fixture(autouse=True)
def _default_raw_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.raw_exists",
        lambda _symbol: True,
    )


def _patch_score(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[list[str]],
) -> None:
    def fake_score(candidates: list[str]) -> list[EmergingLeaderEvaluation]:
        captured.append(candidates)
        return [_evaluation(symbol) for symbol in candidates]

    monkeypatch.setattr(
        "app.services.emerging_leaders_precompute_service.score_emerging_leader_candidates",
        fake_score,
    )


def test_full_universe_precompute_does_not_apply_500_cap_by_default(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.delenv("EMERGING_LEADERS_PRECOMPUTE_MAX_UNIVERSE", raising=False)
    captured: list[list[str]] = []
    _patch_score(monkeypatch, captured)
    symbols = [f"S{i:03d}" for i in range(600)]

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore(symbols),
        snapshot_store=snapshot_store,
    )

    assert result["status"] == "completed"
    assert result["candidates_scanned"] == 600
    assert len(captured[0]) == 600


def test_emergency_cap_only_applies_when_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.setenv("EMERGING_LEADERS_PRECOMPUTE_MAX_UNIVERSE", "10")
    captured: list[list[str]] = []
    _patch_score(monkeypatch, captured)

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore([f"S{i:03d}" for i in range(50)]),
        snapshot_store=snapshot_store,
    )

    assert result["status"] == "completed"
    assert result["emergency_cap"] == 10
    assert result["candidates_scanned"] == 10
    assert len(captured[0]) == 10


def test_top_movers_are_excluded_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.delenv("EMERGING_LEADERS_PRECOMPUTE_MAX_UNIVERSE", raising=False)
    captured: list[list[str]] = []
    _patch_score(monkeypatch, captured)

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore(
            ["AAA", "BBB", "CCC"],
            top_movers=["AAA"],
            run_id="run-1",
        ),
        snapshot_store=snapshot_store,
    )

    assert result["excluded_top_movers"] == 1
    assert captured[0] == ["BBB", "CCC"]


def test_missing_ohlcv_is_excluded_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.raw_exists",
        lambda symbol: symbol != "BBB",
    )
    captured: list[list[str]] = []
    _patch_score(monkeypatch, captured)

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore(["AAA", "BBB", "CCC"]),
        snapshot_store=snapshot_store,
    )

    assert result["symbols_with_data"] == 2
    assert result["candidates_scanned"] == 2
    assert captured[0] == ["AAA", "CCC"]


def test_precompute_stores_top_100_results(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    captured: list[list[str]] = []
    _patch_score(monkeypatch, captured)

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore([f"S{i:03d}" for i in range(120)]),
        snapshot_store=snapshot_store,
    )

    run = snapshot_store.get_run(result["run_id"])
    assert run is not None
    assert run.status == "completed"
    assert run.evaluations_computed == 120
    rows = snapshot_store.list_results(result["run_id"])
    assert len(rows) == 100
    assert rows[0]["rank"] == 1
    assert rows[0]["why_it_ranks"].endswith("is tightening.")


def test_failed_precompute_records_status_and_error(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: EmergingLeadersStore,
) -> None:
    def fail_score(_candidates: list[str]) -> list[EmergingLeaderEvaluation]:
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(
        "app.services.emerging_leaders_precompute_service.score_emerging_leader_candidates",
        fail_score,
    )

    result = precompute_emerging_leaders_snapshot(
        ranking_store=_FakeRankingStore(["AAA"]),
        snapshot_store=snapshot_store,
    )

    assert result["status"] == "failed"
    latest = snapshot_store.latest_run()
    assert latest is not None
    assert latest.status == "failed"
    assert latest.error_message == "scoring exploded"


def test_existing_emerging_leaders_live_scan_behavior_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_collect():
        return ([_evaluation("AAA")], "2026-06-05", 1, 1, 0)

    monkeypatch.setattr(
        "app.services.emerging_leaders_service.collect_qualifying_emerging_leader_evaluations",
        fake_collect,
    )

    response = build_emerging_leaders_live(limit=20)

    assert response.as_of_date == "2026-06-05"
    assert response.universe_scanned == 1
    assert response.evaluations_computed == 1
    assert [item.symbol for item in response.items] == ["AAA"]
