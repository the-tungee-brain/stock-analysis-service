from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.service_dependencies import get_momentum_breakout_scanner_service
from app.main import app
from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_scan_models import MomentumBreakoutScanResponse
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
    _ScanCandidate,
)
from app.services.strategy.momentum_breakout_snapshot_serving_service import (
    MomentumBreakoutSnapshotServingService,
    MomentumBreakoutSnapshotUnavailableError,
)
from app.storage.momentum_breakout_scan_store import MomentumBreakoutScanStore


def _gate(*, allowed: bool = True) -> AlertRiskGateResultDto:
    return AlertRiskGateResultDto(
        allowed=allowed,
        action="ALLOW" if allowed else "BLOCK",
        reasons=[] if allowed else ["blocked"],
        recommendedPositionRiskPct=0.01 if allowed else 0.0,
        alertPriority="HIGH" if allowed else "LOW",
    )


def _candidate(
    symbol: str,
    *,
    setup_score: float = 80.0,
    profit_factor: float | None = 1.5,
    total_trades: int | None = 25,
    stop_distance_pct: float = 5.0,
    allowed: bool = True,
) -> _ScanCandidate:
    return _ScanCandidate(
        symbol=symbol,
        entry_price=100.0,
        stop_price=100.0 - stop_distance_pct,
        target_price=112.0,
        risk_reward=2.4,
        historical_win_rate=0.55,
        historical_profit_factor=profit_factor,
        historical_total_trades=total_trades,
        setup_score=setup_score,
        stop_distance_pct=stop_distance_pct,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="RISK_ON",
        risk_gate=_gate(allowed=allowed),
    )


def _empty_response() -> MomentumBreakoutScanResponse:
    return MomentumBreakoutScanResponse(
        scanTime="2026-06-05T00:00:00+00:00",
        totalSymbolsScanned=0,
        validSetupsFound=0,
        tradableCandidatesFound=0,
        blockedCandidatesCount=0,
        candidatesFound=0,
        candidates=[],
    )


def _result_rows(candidates: list[_ScanCandidate]) -> list[dict]:
    rows: list[dict] = []
    for rank, candidate in enumerate(candidates, start=1):
        dto = MomentumBreakoutScannerService._to_dto(candidate)  # noqa: SLF001
        payload = dto.model_dump(mode="json", by_alias=False)
        rows.append(
            {
                "rank": rank,
                "symbol": payload["symbol"],
                "entry_price": payload["entry_price"],
                "stop_price": payload["stop_price"],
                "target_price": payload["target_price"],
                "risk_reward": payload["risk_reward"],
                "historical_win_rate": payload["historical_win_rate"],
                "historical_profit_factor": payload["historical_profit_factor"],
                "historical_total_trades": payload["historical_total_trades"],
                "setup_score": payload["setup_score"],
                "stop_distance_pct": payload["stop_distance_pct"],
                "volume_ratio": payload["volume_ratio"],
                "rs_percentile": payload["rs_percentile"],
                "market_regime": payload["market_regime"],
                "risk_gate": payload["risk_gate"],
            }
        )
    return rows


def _complete_run(
    store: MomentumBreakoutScanStore,
    *,
    run_id: str,
    generated_at: str,
    candidates: list[_ScanCandidate],
    symbols_scanned: int = 1000,
) -> None:
    store.start_run(run_id=run_id, generated_at=generated_at)
    store.complete_run(
        run_id=run_id,
        as_of_date="2026-06-05",
        generated_at=generated_at,
        ranking_run_id="rank-run",
        ranking_snapshot_id="rank-snapshot",
        universe_source="daily_ranking_results",
        selection_method="ranking_score",
        total_ranked_symbols=symbols_scanned,
        total_eligible_symbols=symbols_scanned,
        symbols_scanned=symbols_scanned,
        excluded_by_cap=0,
        valid_setups_found=len(candidates),
        tradable_candidates_found=0,
        blocked_candidates_count=0,
        duration_ms=1234,
        results=_result_rows(candidates),
    )


@pytest.fixture
def snapshot_store(tmp_path: Path) -> MomentumBreakoutScanStore:
    return MomentumBreakoutScanStore(tmp_path / "ranking.db")


@pytest.fixture
def scanner() -> MomentumBreakoutScannerService:
    return MomentumBreakoutScannerService()


def test_latest_completed_snapshot_used_and_failed_running_ignored(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")
    old_time = "2026-06-05T01:00:00+00:00"
    latest_time = "2026-06-05T02:00:00+00:00"
    _complete_run(
        snapshot_store,
        run_id="completed-old",
        generated_at=old_time,
        candidates=[_candidate("OLD")],
    )
    _complete_run(
        snapshot_store,
        run_id="completed-latest",
        generated_at=latest_time,
        candidates=[_candidate("LATEST")],
    )
    snapshot_store.start_run(run_id="running-newer", generated_at="2026-06-05T03:00:00+00:00")
    snapshot_store.fail_run(
        run_id="running-newer",
        error_message="failed after start",
        duration_ms=5,
    )
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    response = service.scan()

    assert response.scan_time == latest_time
    assert [candidate.symbol for candidate in response.candidates] == ["LATEST"]


def test_snapshot_filters_match_live_scanner_behavior(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")
    candidates = [
        _candidate("OK", setup_score=90.0),
        _candidate("LOWPF", setup_score=80.0, profit_factor=1.0),
        _candidate("LOWTRADES", setup_score=70.0, total_trades=10),
        _candidate("WIDESTOP", setup_score=60.0, stop_distance_pct=9.0),
        _candidate("RISK", setup_score=50.0, allowed=False),
    ]
    _complete_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T02:00:00+00:00",
        candidates=candidates,
        symbols_scanned=500,
    )
    monkeypatch.setattr(scanner, "resolve_symbol_list", lambda _symbols: ["X"] * 500)
    monkeypatch.setattr(scanner, "_collect_candidates", lambda _symbols: candidates)
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    live = scanner.scan(
        symbols=None,
        limit=10,
        tradable_only=True,
        min_historical_profit_factor=1.2,
        min_historical_trades=20,
        max_stop_distance_pct=8.0,
    )
    snapshot = service.scan(
        limit=10,
        tradable_only=True,
        min_historical_profit_factor=1.2,
        min_historical_trades=20,
        max_stop_distance_pct=8.0,
    )

    assert snapshot.total_symbols_scanned == live.total_symbols_scanned
    assert snapshot.valid_setups_found == live.valid_setups_found
    assert snapshot.tradable_candidates_found == live.tradable_candidates_found
    assert snapshot.blocked_candidates_count == live.blocked_candidates_count
    assert snapshot.candidates_found == live.candidates_found
    assert snapshot.model_dump(mode="json", by_alias=True).keys() == live.model_dump(
        mode="json",
        by_alias=True,
    ).keys()
    assert [candidate.symbol for candidate in snapshot.candidates] == ["OK"]


def test_snapshot_limit_applies_after_selected_pool(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")
    _complete_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T02:00:00+00:00",
        candidates=[_candidate("A"), _candidate("B"), _candidate("C")],
    )
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    response = service.scan(limit=2, tradable_only=False)

    assert response.valid_setups_found == 3
    assert response.candidates_found == 2
    assert [candidate.symbol for candidate in response.candidates] == ["A", "B"]


def test_no_snapshot_raises_unavailable_in_precomputed_mode(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    with pytest.raises(MomentumBreakoutSnapshotUnavailableError):
        service.scan()


def test_live_emergency_calls_live_scanner(
    snapshot_store: MomentumBreakoutScanStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "live_emergency")
    scanner = MomentumBreakoutScannerService()
    calls: list[dict] = []

    def fake_scan(**kwargs):
        calls.append(kwargs)
        return _empty_response()

    monkeypatch.setattr(scanner, "scan", fake_scan)
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    service.scan(limit=7)

    assert calls == [
        {
            "limit": 7,
            "tradable_only": False,
            "min_historical_profit_factor": 1.2,
            "min_historical_trades": 20,
            "max_stop_distance_pct": 8.0,
        }
    ]


def test_precomputed_with_live_fallback_only_falls_back_without_snapshot(
    snapshot_store: MomentumBreakoutScanStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed_with_live_fallback")
    scanner = MomentumBreakoutScannerService()
    calls = 0

    def fake_scan(**_kwargs):
        nonlocal calls
        calls += 1
        return _empty_response()

    monkeypatch.setattr(scanner, "scan", fake_scan)
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    service.scan()
    assert calls == 1

    _complete_run(
        snapshot_store,
        run_id="completed",
        generated_at="2026-06-05T02:00:00+00:00",
        candidates=[_candidate("SNAP")],
    )
    service.scan()
    assert calls == 1


def test_stale_snapshot_logs_warning_but_returns_response(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")
    monkeypatch.setenv("MB_SCAN_MAX_SNAPSHOT_AGE_HOURS", "1")
    old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _complete_run(
        snapshot_store,
        run_id="stale",
        generated_at=old,
        candidates=[_candidate("STALE")],
    )
    service = MomentumBreakoutSnapshotServingService(
        store=snapshot_store,
        scanner=scanner,
    )

    response = service.scan()

    assert response.candidates[0].symbol == "STALE"
    assert "serving stale snapshot" in caplog.text


def test_scan_route_explicit_symbols_stays_live(
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")
    calls: list[dict] = []

    def fake_scan(**kwargs):
        calls.append(kwargs)
        return _empty_response()

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    monkeypatch.setattr(scanner, "scan", fake_scan)
    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_scanner_service] = lambda: scanner
    client = TestClient(app)
    try:
        response = client.get(
            "/api/v1/strategy/momentum-breakout/scan",
            params={"symbols": "AAPL", "limit": 3},
        )
        assert response.status_code == 200
        assert calls[0]["symbols"] == "AAPL"
        assert calls[0]["limit"] == 3
    finally:
        app.dependency_overrides.clear()


def test_scan_route_no_snapshot_returns_503(
    snapshot_store: MomentumBreakoutScanStore,
    scanner: MomentumBreakoutScannerService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_ALERTS_ENABLED", "true")
    monkeypatch.setenv("MB_SCAN_SERVING_MODE", "precomputed")

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    old_service = getattr(app.state, "momentum_breakout_snapshot_serving_service", None)
    app.state.momentum_breakout_snapshot_serving_service = (
        MomentumBreakoutSnapshotServingService(
            store=snapshot_store,
            scanner=scanner,
        )
    )
    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_momentum_breakout_scanner_service] = lambda: scanner
    client = TestClient(app)
    try:
        response = client.get("/api/v1/strategy/momentum-breakout/scan")
        assert response.status_code == 503
        assert "precomputed snapshot is not available" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
        if old_service is None:
            delattr(app.state, "momentum_breakout_snapshot_serving_service")
        else:
            app.state.momentum_breakout_snapshot_serving_service = old_service
