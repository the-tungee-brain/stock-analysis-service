"""Momentum Breakout scan universe selection and staleness."""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services.strategy.momentum_breakout_scan_universe import (
    STALE_RANKING_WARNING,
    ScanUniverseOrder,
    build_production_scan_universe,
    is_ranking_output_stale,
    load_production_scan_symbol_list,
    scan_universe_order,
    sort_universe_members,
)
from ranking_pipeline.storage.sqlite import (
    LatestRankingRunMeta,
    RankingResultRecord,
    UniverseMemberRecord,
)

_EASTERN = ZoneInfo("America/New_York")


def _member(
    symbol: str,
    *,
    adv: float | None = None,
    market_cap: float | None = None,
    ranking_score: float | None = None,
) -> UniverseMemberRecord:
    return UniverseMemberRecord(
        symbol=symbol,
        avg_dollar_volume_20d=adv,
        market_cap=market_cap,
        ranking_score=ranking_score,
    )


def test_scan_universe_order_defaults_to_ranking_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MB_SCAN_UNIVERSE_ORDER", raising=False)
    assert scan_universe_order() == ScanUniverseOrder.RANKING_SCORE


def test_sort_universe_members_liquidity_not_alphabetical() -> None:
    members = [
        _member("AAA", adv=1e6),
        _member("ZZZ", adv=9e9),
    ]
    ordered = [m.symbol for m in sort_universe_members(members, ScanUniverseOrder.LIQUIDITY)]
    assert ordered == ["ZZZ", "AAA"]


def test_is_ranking_stale_when_missing_run() -> None:
    assert is_ranking_output_stale(None, total_ranked=0) is True


def test_is_ranking_stale_before_open_without_today_run() -> None:
    run = LatestRankingRunMeta(
        run_id="run-old",
        as_of_date="2026-06-05",
        created_at="2026-06-05T12:00:00+00:00",
        universe_snapshot_id="snap",
        symbol_count=100,
    )
    # Monday 2026-06-08 08:00 ET (before open); run only from prior Friday calendar day.
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    assert is_ranking_output_stale(run, now=now, total_ranked=100) is True


def test_is_ranking_fresh_when_run_created_same_day_before_open() -> None:
    run = LatestRankingRunMeta(
        run_id="run-today",
        as_of_date="2026-06-05",
        created_at="2026-06-08T10:00:00+00:00",
        universe_snapshot_id="snap",
        symbol_count=100,
    )
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    assert is_ranking_output_stale(run, now=now, total_ranked=100) is False


def test_daily_ranking_order_beats_alphabetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_UNIVERSE_ORDER", "ranking_score")
    monkeypatch.setenv("MB_SCAN_MAX_UNIVERSE", "10")

    run = LatestRankingRunMeta(
        run_id="run-1",
        as_of_date="2026-06-05",
        created_at="2026-06-08T10:00:00+00:00",
        universe_snapshot_id="snap-1",
        symbol_count=3,
    )

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "snap-1"

        def load_passed_universe_members(
            self, snapshot_id: str | None = None, *, ranking_run_id: str | None = None
        ) -> list[UniverseMemberRecord]:
            return [
                _member("AAA", adv=1e9),
                _member("ZZZ", adv=1e6),
                _member("MMM", adv=5e8),
            ]

        def get_latest_ranking_run(self) -> LatestRankingRunMeta:
            return run

        def count_ranking_results(self, run_id: str) -> int:
            assert run_id == "run-1"
            return 3

        def load_ranking_results_ordered(self, run_id: str) -> list[RankingResultRecord]:
            return [
                RankingResultRecord("ZZZ", 1, 9.5, 1.0),
                RankingResultRecord("MMM", 2, 8.0, 0.8),
                RankingResultRecord("AAA", 3, 1.0, 0.2),
            ]

    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.open_store",
        lambda _cfg: _FakeStore(),
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.raw_exists",
        lambda _symbol: True,
    )

    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    symbols = load_production_scan_symbol_list(
        max_symbols=10, now=now, store=_FakeStore()
    )
    assert symbols == ["ZZZ", "MMM", "AAA"]
    assert symbols[0] != "AAA"


def test_cap_excludes_lower_ranked_not_alphabetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_MAX_UNIVERSE", "1")

    run = LatestRankingRunMeta(
        run_id="run-1",
        as_of_date="2026-06-05",
        created_at="2026-06-08T10:00:00+00:00",
        universe_snapshot_id="snap-1",
        symbol_count=2,
    )

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "snap-1"

        def load_passed_universe_members(
            self, snapshot_id: str | None = None, *, ranking_run_id: str | None = None
        ) -> list[UniverseMemberRecord]:
            return [_member("AAA"), _member("ZZZ")]

        def get_latest_ranking_run(self) -> LatestRankingRunMeta:
            return run

        def count_ranking_results(self, run_id: str) -> int:
            return 2

        def load_ranking_results_ordered(self, run_id: str) -> list[RankingResultRecord]:
            return [
                RankingResultRecord("ZZZ", 1, 9.0, 1.0),
                RankingResultRecord("AAA", 2, 1.0, 0.1),
            ]

    store = _FakeStore()
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.open_store",
        lambda _cfg: store,
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.raw_exists",
        lambda _symbol: True,
    )

    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    info = build_production_scan_universe(max_symbols=1, now=now, store=store)
    assert info.first_50_scanned_symbols == ["ZZZ"]
    assert info.top_excluded_sample == ["AAA"]
    assert info.excluded_by_cap == 1


def test_stale_ranking_emits_warning_and_liquidity_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_UNIVERSE_ORDER", "ranking_score")

    run = LatestRankingRunMeta(
        run_id="run-stale",
        as_of_date="2026-06-01",
        created_at="2026-06-01T12:00:00+00:00",
        universe_snapshot_id="snap-1",
        symbol_count=2,
    )

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "snap-1"

        def load_passed_universe_members(
            self, snapshot_id: str | None = None, *, ranking_run_id: str | None = None
        ) -> list[UniverseMemberRecord]:
            return [
                _member("AAA", adv=1e6),
                _member("ZZZ", adv=9e9),
            ]

        def get_latest_ranking_run(self) -> LatestRankingRunMeta:
            return run

        def count_ranking_results(self, run_id: str) -> int:
            return 2

        def load_ranking_results_ordered(self, run_id: str) -> list[RankingResultRecord]:
            return [
                RankingResultRecord("AAA", 1, 9.0, 1.0),
                RankingResultRecord("ZZZ", 2, 1.0, 0.1),
            ]

    store = _FakeStore()
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.open_store",
        lambda _cfg: store,
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.raw_exists",
        lambda _symbol: True,
    )

    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    info = build_production_scan_universe(now=now, store=store)
    assert info.warning == STALE_RANKING_WARNING
    assert info.universe_source == "universe_members_liquidity_fallback"
    assert info.first_50_scanned_symbols == ["ZZZ", "AAA"]


def test_liquidity_config_skips_ranking_even_when_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MB_SCAN_UNIVERSE_ORDER", "liquidity")

    run = LatestRankingRunMeta(
        run_id="run-1",
        as_of_date="2026-06-05",
        created_at="2026-06-08T10:00:00+00:00",
        universe_snapshot_id="snap-1",
        symbol_count=2,
    )

    class _FakeStore:
        def active_snapshot_id(self) -> str:
            return "snap-1"

        def load_passed_universe_members(
            self, snapshot_id: str | None = None, *, ranking_run_id: str | None = None
        ) -> list[UniverseMemberRecord]:
            return [
                _member("AAA", adv=1e6),
                _member("ZZZ", adv=9e9),
            ]

        def get_latest_ranking_run(self) -> LatestRankingRunMeta:
            return run

        def count_ranking_results(self, run_id: str) -> int:
            return 2

        def load_ranking_results_ordered(self, run_id: str) -> list[RankingResultRecord]:
            raise AssertionError("should not load ranking rows when order=liquidity")

    store = _FakeStore()
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.open_store",
        lambda _cfg: store,
    )
    monkeypatch.setattr(
        "app.services.strategy.momentum_breakout_scan_universe.raw_exists",
        lambda _symbol: True,
    )

    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)
    info = build_production_scan_universe(now=now, store=store)
    assert info.warning is None
    assert info.universe_source == "universe_members_liquidity"
    assert info.first_50_scanned_symbols == ["ZZZ", "AAA"]
