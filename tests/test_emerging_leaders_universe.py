"""Emerging Leaders production universe selection."""

from __future__ import annotations

import pytest

from app.services.emerging_leaders_evaluations import (
    select_emerging_leader_candidates,
)
from ranking_pipeline.storage.sqlite import (
    LatestRankingRunMeta,
    RankingResultRecord,
    UniverseMemberRecord,
)


def _member(
    symbol: str,
    *,
    adv: float | None = None,
    market_cap: float | None = None,
) -> UniverseMemberRecord:
    return UniverseMemberRecord(
        symbol=symbol,
        avg_dollar_volume_20d=adv,
        market_cap=market_cap,
        ranking_score=None,
    )


def _run() -> LatestRankingRunMeta:
    return LatestRankingRunMeta(
        run_id="run-1",
        as_of_date="2026-06-05",
        created_at="2026-06-05T12:00:00+00:00",
        universe_snapshot_id="snap-1",
        symbol_count=3,
    )


class _FakeStore:
    def __init__(
        self,
        *,
        members: list[UniverseMemberRecord],
        ranking_rows: list[RankingResultRecord] | None = None,
        latest_run: LatestRankingRunMeta | None = None,
    ) -> None:
        self.members = members
        self.ranking_rows = ranking_rows or []
        self.latest_run = latest_run

    def active_snapshot_id(self) -> str:
        return "snap-1"

    def load_passed_universe_members(
        self,
        snapshot_id: str | None = None,
    ) -> list[UniverseMemberRecord]:
        assert snapshot_id == "snap-1"
        return self.members

    def get_latest_ranking_run(self) -> LatestRankingRunMeta | None:
        return self.latest_run

    def count_ranking_results(self, run_id: str) -> int:
        assert self.latest_run is not None
        assert run_id == self.latest_run.run_id
        return len(self.ranking_rows)

    def load_ranking_results_ordered(
        self,
        run_id: str,
    ) -> list[RankingResultRecord]:
        assert self.latest_run is not None
        assert run_id == self.latest_run.run_id
        return self.ranking_rows


@pytest.fixture(autouse=True)
def _raw_data_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.raw_exists",
        lambda _symbol: True,
    )


def test_candidate_cap_applies_after_liquidity_order_not_alphabetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: True,
    )
    store = _FakeStore(
        members=[
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("ZZZ", adv=9_000_000_000, market_cap=1_000_000_000),
            _member("MMM", adv=5_000_000_000, market_cap=1_000_000_000),
        ],
        latest_run=None,
    )

    candidates, symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=1,
        top_mover_symbols=set(),
    )

    assert candidates == ["ZZZ"]
    assert symbols_with_data == 3


def test_fresh_ranking_order_beats_liquidity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: False,
    )
    store = _FakeStore(
        members=[
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("ZZZ", adv=9_000_000_000, market_cap=1_000_000_000),
            _member("MMM", adv=5_000_000_000, market_cap=1_000_000_000),
        ],
        ranking_rows=[
            RankingResultRecord("AAA", 1, 99.0, 0.99),
            RankingResultRecord("ZZZ", 2, 80.0, 0.80),
            RankingResultRecord("MMM", 3, 70.0, 0.70),
        ],
        latest_run=_run(),
    )

    candidates, _symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=2,
        top_mover_symbols=set(),
    )

    assert candidates == ["AAA", "ZZZ"]


def test_stale_ranking_falls_back_to_liquidity_and_market_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: True,
    )
    store = _FakeStore(
        members=[
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("BBB", adv=9_000_000_000, market_cap=2_000_000_000),
            _member("ZZZ", adv=9_000_000_000, market_cap=5_000_000_000),
        ],
        ranking_rows=[
            RankingResultRecord("AAA", 1, 99.0, 0.99),
            RankingResultRecord("BBB", 2, 80.0, 0.80),
            RankingResultRecord("ZZZ", 3, 70.0, 0.70),
        ],
        latest_run=_run(),
    )

    candidates, _symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=3,
        top_mover_symbols=set(),
    )

    assert candidates == ["ZZZ", "BBB", "AAA"]


def test_missing_ranking_falls_back_to_liquidity_not_alphabetical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: True,
    )
    store = _FakeStore(
        members=[
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("ZZZ", adv=9_000_000_000, market_cap=1_000_000_000),
        ],
        latest_run=None,
    )

    candidates, _symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=2,
        top_mover_symbols=set(),
    )

    assert candidates == ["ZZZ", "AAA"]


def test_top_movers_are_excluded_before_candidate_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: False,
    )
    store = _FakeStore(
        members=[
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("ZZZ", adv=9_000_000_000, market_cap=1_000_000_000),
            _member("MMM", adv=5_000_000_000, market_cap=1_000_000_000),
        ],
        ranking_rows=[
            RankingResultRecord("AAA", 1, 99.0, 0.99),
            RankingResultRecord("ZZZ", 2, 80.0, 0.80),
            RankingResultRecord("MMM", 3, 70.0, 0.70),
        ],
        latest_run=_run(),
    )

    candidates, _symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=1,
        top_mover_symbols={"AAA"},
    )

    assert candidates == ["ZZZ"]


def test_symbol_alphabetical_order_is_only_tie_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.services.emerging_leaders_evaluations.is_ranking_output_stale",
        lambda _run, *, total_ranked: True,
    )
    store = _FakeStore(
        members=[
            _member("ZZZ", adv=1_000_000, market_cap=1_000_000_000),
            _member("AAA", adv=1_000_000, market_cap=1_000_000_000),
            _member("MMM", adv=2_000_000, market_cap=1_000_000_000),
        ],
        latest_run=None,
    )

    candidates, _symbols_with_data = select_emerging_leader_candidates(
        store,
        max_universe=3,
        top_mover_symbols=set(),
    )

    assert candidates == ["MMM", "AAA", "ZZZ"]
