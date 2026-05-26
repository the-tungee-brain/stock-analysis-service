from __future__ import annotations

from app.models.company_research_models import EtfHoldingItem

PIOTROSKI_MAX = 9
ALTMAN_DISTRESS = 1.81
ALTMAN_SAFE = 2.99
DEFAULT_QUALITY_LIMIT = 5


def normalize_piotroski_score(value: int | None) -> float | None:
    if value is None:
        return None
    clamped = max(0, min(int(value), PIOTROSKI_MAX))
    return clamped / PIOTROSKI_MAX


def normalize_altman_z_score(value: float | None) -> float | None:
    if value is None:
        return None
    z = float(value)
    if z < ALTMAN_DISTRESS:
        return max(0.0, (z / ALTMAN_DISTRESS) * 0.35)
    if z < ALTMAN_SAFE:
        span = ALTMAN_SAFE - ALTMAN_DISTRESS
        return 0.35 + ((z - ALTMAN_DISTRESS) / span) * 0.35
    return min(0.7 + ((z - ALTMAN_SAFE) / 10.0) * 0.3, 1.0)


def compute_quality_score(
    piotroski_f: int | None,
    altman_z: float | None,
) -> float | None:
    parts = [
        score
        for score in (
            normalize_piotroski_score(piotroski_f),
            normalize_altman_z_score(altman_z),
        )
        if score is not None
    ]
    if not parts:
        return None
    return sum(parts) / len(parts)


def rank_etf_holdings_by_quality(
    holdings: list[EtfHoldingItem],
    *,
    limit: int = DEFAULT_QUALITY_LIMIT,
) -> tuple[list[EtfHoldingItem], list[EtfHoldingItem]]:
    scored: list[EtfHoldingItem] = []
    for holding in holdings:
        if holding.quality_score is None:
            continue
        scored.append(holding)

    if not scored:
        return [], []

    ranked = sorted(
        scored,
        key=lambda item: (
            item.quality_score or 0.0,
            item.piotroski_f or 0,
            item.altman_z or 0.0,
            item.weight_pct,
        ),
        reverse=True,
    )
    resolved_limit = max(1, limit)
    strongest = ranked[:resolved_limit]
    weakest = list(reversed(ranked[-resolved_limit:]))
    return strongest, weakest
