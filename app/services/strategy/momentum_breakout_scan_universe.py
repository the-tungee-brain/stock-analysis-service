"""Production scan universe: daily ranking output with liquidity/alphabetical fallbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum

from data.store import raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import (
    LatestRankingRunMeta,
    RankingResultRecord,
    RankingStore,
    UniverseMemberRecord,
    open_store,
)
from trade_planner.alerts.market_calendar import (
    _EASTERN,
    is_before_regular_session_open,
    latest_completed_bar_trading_day,
    trading_days_apart,
)

from app.models.momentum_breakout_scan_models import MomentumBreakoutUniverseResponse

TOP_EXCLUDED_SAMPLE_SIZE = 20
STALE_RANKING_WARNING = (
    "Ranking output is stale; scanner is using fallback universe."
)

UNIVERSE_SOURCE_DAILY_RANKING = "daily_ranking_results"
UNIVERSE_SOURCE_LIQUIDITY_FALLBACK = "universe_members_liquidity_fallback"
UNIVERSE_SOURCE_ALPHABETICAL_FALLBACK = "universe_members_alphabetical_fallback"
UNIVERSE_SOURCE_LIQUIDITY_CONFIG = "universe_members_liquidity"
UNIVERSE_SOURCE_MARKET_CAP_CONFIG = "universe_members_market_cap"
UNIVERSE_SOURCE_ALPHABETICAL_CONFIG = "universe_members_alphabetical"


class ScanUniverseOrder(str, Enum):
    RANKING_SCORE = "ranking_score"
    LIQUIDITY = "liquidity"
    MARKET_CAP = "market_cap"
    ALPHABETICAL = "alphabetical"


_SELECTION_METHOD_LABELS: dict[ScanUniverseOrder, str] = {
    ScanUniverseOrder.RANKING_SCORE: (
        "ranking_score: daily ranking_results ORDER BY rank ASC (final_score DESC)"
    ),
    ScanUniverseOrder.LIQUIDITY: (
        "liquidity: avg_dollar_volume_20d DESC, market_cap DESC, symbol ASC"
    ),
    ScanUniverseOrder.MARKET_CAP: (
        "market_cap: market_cap DESC, avg_dollar_volume_20d DESC, symbol ASC"
    ),
    ScanUniverseOrder.ALPHABETICAL: "alphabetical: symbol ASC",
}


@dataclass(frozen=True, slots=True)
class _UniverseBuild:
    symbols: list[str]
    universe_source: str
    selection_method: str
    ranking_run_id: str | None
    ranking_snapshot_id: str | None
    ranking_generated_at: str | None
    total_ranked_symbols: int
    warning: str | None


def scan_universe_order() -> ScanUniverseOrder:
    raw = os.environ.get("MB_SCAN_UNIVERSE_ORDER", ScanUniverseOrder.RANKING_SCORE.value)
    try:
        return ScanUniverseOrder(raw.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(order.value for order in ScanUniverseOrder)
        raise ValueError(
            f"Invalid MB_SCAN_UNIVERSE_ORDER={raw!r}; expected one of: {allowed}"
        ) from exc


def max_scan_universe() -> int:
    return int(os.environ.get("MB_SCAN_MAX_UNIVERSE", "500"))


def _metric_or_sentinel(value: float | None, *, missing: float) -> float:
    return float(value) if value is not None else missing


def _has_liquidity_metrics(members: list[UniverseMemberRecord]) -> bool:
    return any(
        member.avg_dollar_volume_20d is not None or member.market_cap is not None
        for member in members
    )


def sort_universe_members(
    members: list[UniverseMemberRecord],
    order: ScanUniverseOrder,
) -> list[UniverseMemberRecord]:
    if order == ScanUniverseOrder.ALPHABETICAL:
        return sorted(members, key=lambda member: member.symbol)

    if order == ScanUniverseOrder.MARKET_CAP:
        return sorted(
            members,
            key=lambda member: (
                -_metric_or_sentinel(member.market_cap, missing=-1.0),
                -_metric_or_sentinel(member.avg_dollar_volume_20d, missing=-1.0),
                member.symbol,
            ),
        )

    if order == ScanUniverseOrder.RANKING_SCORE:
        return sorted(
            members,
            key=lambda member: (
                -_metric_or_sentinel(member.ranking_score, missing=float("-inf")),
                -_metric_or_sentinel(member.avg_dollar_volume_20d, missing=-1.0),
                -_metric_or_sentinel(member.market_cap, missing=-1.0),
                member.symbol,
            ),
        )

    return sorted(
        members,
        key=lambda member: (
            -_metric_or_sentinel(member.avg_dollar_volume_20d, missing=-1.0),
            -_metric_or_sentinel(member.market_cap, missing=-1.0),
            member.symbol,
        ),
    )


def _parse_run_created_at(created_at: str) -> datetime:
    instant = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        return instant.replace(tzinfo=timezone.utc)
    return instant.astimezone(timezone.utc)


def is_ranking_output_stale(
    run: LatestRankingRunMeta | None,
    *,
    now: datetime | None = None,
    total_ranked: int = 0,
) -> bool:
    """True when daily ranking is missing, empty, too old, or not refreshed before today's open."""
    if run is None or total_ranked <= 0:
        return True

    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)

    expected_bar_day = latest_completed_bar_trading_day(instant)
    try:
        as_of_day = date.fromisoformat(run.as_of_date)
    except ValueError:
        return True

    if trading_days_apart(as_of_day, expected_bar_day) > 1:
        return True

    if is_before_regular_session_open(instant):
        created_et = _parse_run_created_at(run.created_at).astimezone(_EASTERN)
        et_now = instant.astimezone(_EASTERN)
        if created_et.date() < et_now.date():
            return True

    return False


def _order_from_daily_ranking(
    store: RankingStore,
    run: LatestRankingRunMeta,
    members: list[UniverseMemberRecord],
) -> list[str]:
    ranked_rows = store.load_ranking_results_ordered(run.run_id)
    universe_symbols = {member.symbol for member in members}
    ordered: list[str] = []
    seen: set[str] = set()
    for row in ranked_rows:
        sym = row.symbol
        if sym not in universe_symbols or sym in seen:
            continue
        if not raw_exists(sym):
            continue
        seen.add(sym)
        ordered.append(sym)

    tail = [
        member
        for member in members
        if member.symbol not in seen and raw_exists(member.symbol)
    ]
    if tail:
        ordered.extend(member.symbol for member in sort_universe_members(tail, ScanUniverseOrder.LIQUIDITY))
    return ordered


def _order_from_members(
    members: list[UniverseMemberRecord],
    order: ScanUniverseOrder,
) -> tuple[list[str], str, str]:
    eligible = [member for member in members if raw_exists(member.symbol)]
    if not eligible:
        return [], UNIVERSE_SOURCE_ALPHABETICAL_FALLBACK, _SELECTION_METHOD_LABELS[ScanUniverseOrder.ALPHABETICAL]

    if not _has_liquidity_metrics(eligible):
        ranked = sort_universe_members(eligible, ScanUniverseOrder.ALPHABETICAL)
        return (
            [member.symbol for member in ranked],
            UNIVERSE_SOURCE_ALPHABETICAL_FALLBACK,
            _SELECTION_METHOD_LABELS[ScanUniverseOrder.ALPHABETICAL],
        )

    if order == ScanUniverseOrder.MARKET_CAP:
        source = UNIVERSE_SOURCE_MARKET_CAP_CONFIG
    elif order == ScanUniverseOrder.ALPHABETICAL:
        source = UNIVERSE_SOURCE_ALPHABETICAL_CONFIG
    elif order == ScanUniverseOrder.LIQUIDITY:
        source = UNIVERSE_SOURCE_LIQUIDITY_CONFIG
    else:
        source = UNIVERSE_SOURCE_LIQUIDITY_FALLBACK

    ranked = sort_universe_members(eligible, order)
    return [member.symbol for member in ranked], source, _SELECTION_METHOD_LABELS[order]


def build_scan_universe_symbols(
    *,
    max_symbols: int | None = None,
    order: ScanUniverseOrder | None = None,
    now: datetime | None = None,
    store: RankingStore | None = None,
) -> _UniverseBuild:
    order_mode = order or scan_universe_order()
    ranking_store = store or open_store(default_config())
    snapshot_id = ranking_store.active_snapshot_id()
    if not snapshot_id:
        raise LookupError("No active ranking universe snapshot")

    members = ranking_store.load_passed_universe_members(snapshot_id)
    if not members:
        raise LookupError("No active ranking universe")

    latest_run = ranking_store.get_latest_ranking_run()
    total_ranked = (
        ranking_store.count_ranking_results(latest_run.run_id)
        if latest_run is not None
        else 0
    )
    stale = is_ranking_output_stale(latest_run, now=now, total_ranked=total_ranked)
    warning = STALE_RANKING_WARNING if stale and order_mode == ScanUniverseOrder.RANKING_SCORE else None

    if (
        order_mode == ScanUniverseOrder.RANKING_SCORE
        and not stale
        and latest_run is not None
        and total_ranked > 0
    ):
        ordered = _order_from_daily_ranking(ranking_store, latest_run, members)
        build = _UniverseBuild(
            symbols=ordered,
            universe_source=UNIVERSE_SOURCE_DAILY_RANKING,
            selection_method=_SELECTION_METHOD_LABELS[ScanUniverseOrder.RANKING_SCORE],
            ranking_run_id=latest_run.run_id,
            ranking_snapshot_id=latest_run.universe_snapshot_id or snapshot_id,
            ranking_generated_at=latest_run.created_at,
            total_ranked_symbols=total_ranked,
            warning=None,
        )
    elif order_mode == ScanUniverseOrder.RANKING_SCORE and stale:
        symbols, _source, method = _order_from_members(members, ScanUniverseOrder.LIQUIDITY)
        build = _UniverseBuild(
            symbols=symbols,
            universe_source=UNIVERSE_SOURCE_LIQUIDITY_FALLBACK,
            selection_method=method,
            ranking_run_id=latest_run.run_id if latest_run else None,
            ranking_snapshot_id=(
                (latest_run.universe_snapshot_id if latest_run else None) or snapshot_id
            ),
            ranking_generated_at=latest_run.created_at if latest_run else None,
            total_ranked_symbols=total_ranked,
            warning=warning,
        )
    else:
        symbols, source, method = _order_from_members(members, order_mode)
        build = _UniverseBuild(
            symbols=symbols,
            universe_source=source,
            selection_method=method,
            ranking_run_id=latest_run.run_id if latest_run else None,
            ranking_snapshot_id=(
                (latest_run.universe_snapshot_id if latest_run else None) or snapshot_id
            ),
            ranking_generated_at=latest_run.created_at if latest_run else None,
            total_ranked_symbols=total_ranked,
            warning=None,
        )

    return build


def build_production_scan_universe(
    *,
    max_symbols: int | None = None,
    order: ScanUniverseOrder | None = None,
    now: datetime | None = None,
    store: RankingStore | None = None,
) -> MomentumBreakoutUniverseResponse:
    cap = max(1, max_symbols if max_symbols is not None else max_scan_universe())
    build = build_scan_universe_symbols(
        max_symbols=cap, order=order, now=now, store=store
    )
    scanned = build.symbols[:cap]
    excluded = build.symbols[cap:]
    return MomentumBreakoutUniverseResponse(
        universeSource=build.universe_source,
        selectionMethod=build.selection_method,
        rankingRunId=build.ranking_run_id,
        rankingSnapshotId=build.ranking_snapshot_id,
        rankingGeneratedAt=build.ranking_generated_at,
        totalRankedSymbols=build.total_ranked_symbols,
        totalEligibleSymbols=len(build.symbols),
        scanCap=cap,
        symbolsScanned=len(scanned),
        excludedByCap=max(0, len(build.symbols) - len(scanned)),
        first50ScannedSymbols=scanned[:50],
        topExcludedSample=excluded[:TOP_EXCLUDED_SAMPLE_SIZE],
        warning=build.warning,
    )


def load_production_scan_symbol_list(
    *,
    max_symbols: int,
    order: ScanUniverseOrder | None = None,
    now: datetime | None = None,
    store: RankingStore | None = None,
) -> list[str]:
    build = build_scan_universe_symbols(
        max_symbols=max_symbols, order=order, now=now, store=store
    )
    return build.symbols[: max(1, max_symbols)]
