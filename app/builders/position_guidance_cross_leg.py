from __future__ import annotations

from app.builders.position_guidance_loss_severity import (
    equity_loss_severity,
    option_loss_severity,
    verdict_urgency_rank,
)
from app.models.position_guidance_models import PositionGuidanceItem, PositionKind

_LONG_OPTION_KINDS: frozenset[PositionKind] = frozenset({"LONG_CALL", "LONG_PUT"})

_CROSS_LEG_OPTION_SEVERITY_MIN = 70
_CROSS_LEG_EQUITY_SEVERITY_MAX = 25
_RANK_BOOST = 25


def apply_cross_leg_sanity(
    items: list[PositionGuidanceItem],
) -> list[PositionGuidanceItem]:
    """
    After per-leg scoring: flag cross-leg mismatch and boost option attention ranking.
    Does not change deterministic verdicts.
    """
    equity_items = [i for i in items if i.position_kind == "EQUITY_LONG"]
    long_options = [i for i in items if i.position_kind in _LONG_OPTION_KINDS]
    if not equity_items or not long_options:
        return items

    max_equity_sev = max(
        equity_loss_severity(i.open_profit_loss_pct) for i in equity_items
    )
    max_option_sev = max(
        option_loss_severity(i.open_profit_loss_pct, position_kind=i.position_kind)
        for i in long_options
    )
    if max_option_sev < _CROSS_LEG_OPTION_SEVERITY_MIN:
        return items
    if max_equity_sev > _CROSS_LEG_EQUITY_SEVERITY_MAX:
        return items

    max_equity_rank = max(i.relative_risk_rank for i in equity_items)
    max_equity_verdict_rank = max(verdict_urgency_rank(i.verdict) for i in equity_items)

    updated: list[PositionGuidanceItem] = []
    for item in items:
        if item.position_kind not in _LONG_OPTION_KINDS:
            updated.append(item)
            continue
        opt_sev = option_loss_severity(
            item.open_profit_loss_pct, position_kind=item.position_kind
        )
        if opt_sev < _CROSS_LEG_OPTION_SEVERITY_MIN:
            updated.append(item)
            continue

        new_rank = max(
            item.relative_risk_rank,
            max_equity_rank + _RANK_BOOST,
            max_equity_verdict_rank * 20 + 40,
        )
        new_rank = min(100, int(new_rank))

        updated.append(
            item.model_copy(
                update={
                    "relative_risk_rank": new_rank,
                    "cross_leg_sanity": True,
                }
            )
        )
    updated.sort(
        key=lambda row: (-row.relative_risk_rank, -row.urgency, row.display_label)
    )
    return updated
