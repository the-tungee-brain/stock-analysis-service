from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_scan_models import MomentumBreakoutScanCandidateDto
from app.services.strategy.momentum_breakout_precompute_service import (
    precompute_momentum_breakout_scan_snapshot,
)
from app.services.strategy.momentum_breakout_scanner_service import (
    MomentumBreakoutScannerService,
    _ScanCandidate,
)
from app.storage.momentum_breakout_scan_store import MomentumBreakoutScanStore


def _candidate(
    symbol: str,
    *,
    setup_score: float = 80.0,
    profit_factor: float | None = 1.5,
    total_trades: int | None = 25,
    allowed: bool = True,
) -> _ScanCandidate:
    return _ScanCandidate(
        symbol=symbol,
        entry_price=100.0,
        stop_price=95.0,
        target_price=112.0,
        risk_reward=2.4,
        historical_win_rate=0.55,
        historical_profit_factor=profit_factor,
        historical_total_trades=total_trades,
        setup_score=setup_score,
        stop_distance_pct=5.0,
        volume_ratio=2.0,
        rs_percentile=85.0,
        market_regime="BULL",
        risk_gate=AlertRiskGateResultDto(
            allowed=allowed,
            action="ALLOW" if allowed else "BLOCK",
            reasons=[] if allowed else ["blocked"],
            recommendedPositionRiskPct=0.01 if allowed else 0.0,
            alertPriority="HIGH" if allowed else "LOW",
        ),
    )


class _FakeScanner:
    def __init__(self, candidates: list[_ScanCandidate] | None = None) -> None:
        self.candidates = candidates or []
        self.scanned_symbols: list[str] = []

    def _collect_candidates(self, symbols: list[str]) -> list[_ScanCandidate]:
        self.scanned_symbols = list(symbols)
        return list(self.candidates)


class _FailingScanner:
    def _collect_candidates(self, _symbols: list[str]) -> list[_ScanCandidate]:
        raise RuntimeError("scan exploded")


class _FakeRankingStore:
    def get_run_meta(self, run_id: str) -> dict[str, str]:
        assert run_id == "rank-run-1"
        return {"as_of_date": "2026-06-05"}


@pytest.fixture
def snapshot_store(tmp_path: Path) -> MomentumBreakoutScanStore:
    return MomentumBreakoutScanStore(tmp_path / "ranking.db")


@pytest.fixture
def fake_universe(monkeypatch: pytest.MonkeyPatch):
    def install(symbols: list[str]) -> None:
        monkeypatch.setattr(
            "app.services.strategy.momentum_breakout_precompute_service.build_scan_universe_symbols",
            lambda **_kwargs: SimpleNamespace(
                symbols=symbols,
                universe_source="daily_ranking_results",
                selection_method="ranking_score",
                ranking_run_id="rank-run-1",
                ranking_snapshot_id="snap-1",
                total_ranked_symbols=len(symbols),
            ),
        )

    return install


def test_precompute_does_not_apply_mb_scan_max_universe_by_default(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: MomentumBreakoutScanStore,
    fake_universe,
) -> None:
    monkeypatch.setenv("MB_SCAN_MAX_UNIVERSE", "1")
    monkeypatch.delenv("MB_PRECOMPUTE_MAX_UNIVERSE", raising=False)
    symbols = [f"S{i:03d}" for i in range(600)]
    fake_universe(symbols)
    scanner = _FakeScanner([_candidate("AAA")])

    result = precompute_momentum_breakout_scan_snapshot(
        ranking_store=_FakeRankingStore(),
        snapshot_store=snapshot_store,
        scanner=scanner,  # type: ignore[arg-type]
    )

    assert result["status"] == "completed"
    assert result["symbols_scanned"] == 600
    assert result["excluded_by_cap"] == 0
    assert len(scanner.scanned_symbols) == 600


def test_emergency_cap_only_applies_when_explicitly_configured(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: MomentumBreakoutScanStore,
    fake_universe,
) -> None:
    monkeypatch.setenv("MB_PRECOMPUTE_MAX_UNIVERSE", "10")
    symbols = [f"S{i:03d}" for i in range(50)]
    fake_universe(symbols)
    scanner = _FakeScanner([_candidate("AAA")])

    result = precompute_momentum_breakout_scan_snapshot(
        ranking_store=_FakeRankingStore(),
        snapshot_store=snapshot_store,
        scanner=scanner,  # type: ignore[arg-type]
    )

    assert result["status"] == "completed"
    assert result["emergency_cap"] == 10
    assert result["symbols_scanned"] == 10
    assert result["excluded_by_cap"] == 40
    assert scanner.scanned_symbols == symbols[:10]


def test_cli_max_universe_overrides_env_cap(
    monkeypatch: pytest.MonkeyPatch,
    snapshot_store: MomentumBreakoutScanStore,
    fake_universe,
) -> None:
    monkeypatch.setenv("MB_PRECOMPUTE_MAX_UNIVERSE", "10")
    symbols = [f"S{i:03d}" for i in range(50)]
    fake_universe(symbols)
    scanner = _FakeScanner([_candidate("AAA")])

    result = precompute_momentum_breakout_scan_snapshot(
        ranking_store=_FakeRankingStore(),
        snapshot_store=snapshot_store,
        scanner=scanner,  # type: ignore[arg-type]
        max_universe=7,
    )

    assert result["emergency_cap"] == 7
    assert scanner.scanned_symbols == symbols[:7]


def test_failed_precompute_records_status_and_error(
    snapshot_store: MomentumBreakoutScanStore,
    fake_universe,
) -> None:
    fake_universe(["AAA"])

    result = precompute_momentum_breakout_scan_snapshot(
        ranking_store=_FakeRankingStore(),
        snapshot_store=snapshot_store,
        scanner=_FailingScanner(),  # type: ignore[arg-type]
    )

    assert result["status"] == "failed"
    latest = snapshot_store.latest_run()
    assert latest is not None
    assert latest.status == "failed"
    assert latest.error_message == "scan exploded"


def test_stored_results_can_reconstruct_current_dto_shape(
    snapshot_store: MomentumBreakoutScanStore,
    fake_universe,
) -> None:
    fake_universe(["AAA", "BBB"])
    scanner = _FakeScanner(
        [
            _candidate("LOW", setup_score=50.0, profit_factor=3.0),
            _candidate("HIGH", setup_score=90.0, profit_factor=1.1, total_trades=30),
            _candidate("BLOCK", setup_score=80.0, allowed=False),
        ]
    )

    result = precompute_momentum_breakout_scan_snapshot(
        ranking_store=_FakeRankingStore(),
        snapshot_store=snapshot_store,
        scanner=scanner,  # type: ignore[arg-type]
    )

    run = snapshot_store.get_run(result["run_id"])
    assert run is not None
    assert run.status == "completed"
    assert run.valid_setups_found == 3
    assert run.tradable_candidates_found == 1
    assert run.blocked_candidates_count == 2
    rows = snapshot_store.list_results(result["run_id"])
    assert [row["symbol"] for row in rows] == ["HIGH", "BLOCK", "LOW"]

    dto = MomentumBreakoutScanCandidateDto.model_validate(rows[0])
    payload = dto.model_dump(mode="json", by_alias=True)
    assert set(payload) == {
        "symbol",
        "entryPrice",
        "stopPrice",
        "targetPrice",
        "riskReward",
        "historicalWinRate",
        "historicalProfitFactor",
        "historicalTotalTrades",
        "setupScore",
        "stopDistancePct",
        "volumeRatio",
        "rsPercentile",
        "marketRegime",
        "riskGate",
    }
    assert payload["riskGate"]["recommendedPositionRiskPct"] == 0.01


def test_existing_scan_and_explicit_symbols_path_remain_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = MomentumBreakoutScannerService()
    captured: list[list[str]] = []

    def fake_collect(symbols: list[str]) -> list[_ScanCandidate]:
        captured.append(symbols)
        return [_candidate("MSFT")]

    monkeypatch.setattr(scanner, "_collect_candidates", fake_collect)

    response = scanner.scan(symbols="msft,aapl,MSFT", limit=5)

    assert captured == [["MSFT", "AAPL"]]
    assert response.total_symbols_scanned == 2
    assert response.valid_setups_found == 1
    assert [candidate.symbol for candidate in response.candidates] == ["MSFT"]
