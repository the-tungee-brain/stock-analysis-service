from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.builders.emerging_leaders_engine import (
    STAGE_LABELS,
    compression_velocity_label,
)
from app.models.emerging_leaders_models import EmergingLeaderItem, EmergingLeadersResponse
from app.services.emerging_leaders_evaluations import (
    collect_qualifying_emerging_leader_evaluations,
)

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_emerging_leaders(*, limit: int = 20) -> EmergingLeadersResponse:
    (
        evaluations,
        as_of_date,
        candidates_scanned,
        symbols_with_data,
        excluded_top_movers,
    ) = collect_qualifying_emerging_leader_evaluations()
    trimmed = evaluations[: max(1, min(limit, 50))]

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
