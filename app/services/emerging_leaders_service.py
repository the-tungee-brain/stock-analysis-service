from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from app.builders.emerging_leaders_engine import (
    STAGE_LABELS,
    compression_velocity_label,
)
from app.models.emerging_leaders_models import EmergingLeaderItem, EmergingLeadersResponse
from app.services.emerging_leaders_evaluations import (
    collect_qualifying_emerging_leader_evaluations,
)
from app.storage.emerging_leaders_store import (
    EmergingLeadersStore,
    open_emerging_leaders_store,
)

logger = logging.getLogger(__name__)

SERVING_MODE_PRECOMPUTED = "precomputed"
SERVING_MODE_LIVE_EMERGENCY = "live_emergency"
SERVING_MODE_PRECOMPUTED_WITH_LIVE_FALLBACK = "precomputed_with_live_fallback"
SNAPSHOT_UNAVAILABLE_DETAIL = (
    "Emerging Leaders snapshot unavailable; precompute has not completed."
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmergingLeadersSnapshotUnavailable(LookupError):
    pass


def _serving_mode() -> str:
    return os.environ.get(
        "EMERGING_LEADERS_SERVING_MODE",
        SERVING_MODE_PRECOMPUTED,
    ).strip().lower()


def _max_snapshot_age_hours() -> int:
    return int(os.environ.get("EMERGING_LEADERS_MAX_SNAPSHOT_AGE_HOURS", "36"))


def _bounded_limit(limit: int) -> int:
    return max(1, min(limit, 50))


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _warn_if_stale(generated_at: str) -> None:
    max_age_hours = _max_snapshot_age_hours()
    generated = _parse_timestamp(generated_at)
    if generated is None:
        logger.warning(
            "Emerging Leaders snapshot has unparsable generated_at=%r",
            generated_at,
        )
        return
    age_hours = (datetime.now(timezone.utc) - generated).total_seconds() / 3600
    if age_hours > max_age_hours:
        logger.warning(
            "Emerging Leaders snapshot is stale: age_hours=%.2f max_age_hours=%d",
            age_hours,
            max_age_hours,
        )


def build_emerging_leaders_live(*, limit: int = 20) -> EmergingLeadersResponse:
    (
        evaluations,
        as_of_date,
        candidates_scanned,
        symbols_with_data,
        excluded_top_movers,
    ) = collect_qualifying_emerging_leader_evaluations()
    trimmed = evaluations[: _bounded_limit(limit)]

    items = [
        EmergingLeaderItem(
            rank=idx,
            symbol=ev.symbol,
            setup_quality_score=ev.setup_quality_score,
            setup_stage=ev.setup_stage,
            setup_stage_label=STAGE_LABELS[ev.setup_stage],
            compression_velocity=int(ev.components.compression_velocity),
            compression_velocity_label=compression_velocity_label(
                ev.components.compression_velocity
            ),
            why_it_ranks=ev.why_it_ranks,
            positive_factors=ev.positive_factors,
            missing_factors=ev.missing_factors,
            next_confirmation=ev.next_confirmation,
        )
        for idx, ev in enumerate(trimmed, start=1)
    ]

    if not items and symbols_with_data:
        logger.warning(
            "Emerging leaders: 0 list items after scoring (candidates=%d)",
            candidates_scanned,
        )

    return EmergingLeadersResponse(
        as_of_date=as_of_date,
        timestamp=_utc_timestamp(),
        universe_scanned=candidates_scanned,
        symbols_with_data=symbols_with_data,
        evaluations_computed=len(evaluations),
        excluded_top_movers=excluded_top_movers,
        items=items,
    )


def build_emerging_leaders_from_snapshot(
    *,
    limit: int = 20,
    store: EmergingLeadersStore | None = None,
) -> EmergingLeadersResponse:
    snapshot_store = store or open_emerging_leaders_store()
    run = snapshot_store.latest_completed_run()
    if run is None:
        raise EmergingLeadersSnapshotUnavailable(SNAPSHOT_UNAVAILABLE_DETAIL)

    _warn_if_stale(run.generated_at)
    rows = snapshot_store.list_results(run.run_id, limit=_bounded_limit(limit))
    items = [
        EmergingLeaderItem(
            rank=idx,
            symbol=row["symbol"],
            setup_quality_score=row["setup_quality_score"],
            setup_stage=row["setup_stage"],
            setup_stage_label=row["setup_stage_label"],
            compression_velocity=row["compression_velocity"],
            compression_velocity_label=row["compression_velocity_label"],
            why_it_ranks=row["why_it_ranks"],
            positive_factors=row["positive_factors"],
            missing_factors=row["missing_factors"],
            next_confirmation=row["next_confirmation"],
        )
        for idx, row in enumerate(rows, start=1)
    ]
    return EmergingLeadersResponse(
        as_of_date=run.as_of_date,
        timestamp=run.generated_at,
        universe_scanned=run.candidates_scanned,
        symbols_with_data=run.symbols_with_data,
        evaluations_computed=run.evaluations_computed,
        excluded_top_movers=run.excluded_top_movers,
        items=items,
    )


def build_emerging_leaders(*, limit: int = 20) -> EmergingLeadersResponse:
    mode = _serving_mode()
    if mode == SERVING_MODE_LIVE_EMERGENCY:
        logger.warning("Emerging Leaders live emergency serving mode is enabled")
        return build_emerging_leaders_live(limit=limit)

    if mode == SERVING_MODE_PRECOMPUTED_WITH_LIVE_FALLBACK:
        try:
            return build_emerging_leaders_from_snapshot(limit=limit)
        except EmergingLeadersSnapshotUnavailable:
            logger.warning(
                "Emerging Leaders snapshot unavailable; using live fallback"
            )
            return build_emerging_leaders_live(limit=limit)

    if mode != SERVING_MODE_PRECOMPUTED:
        raise ValueError(
            "Invalid EMERGING_LEADERS_SERVING_MODE="
            f"{mode!r}; expected one of: {SERVING_MODE_PRECOMPUTED}, "
            f"{SERVING_MODE_LIVE_EMERGENCY}, "
            f"{SERVING_MODE_PRECOMPUTED_WITH_LIVE_FALLBACK}"
        )

    return build_emerging_leaders_from_snapshot(limit=limit)
