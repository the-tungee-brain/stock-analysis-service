from __future__ import annotations

import re
from collections.abc import Callable

from app.models.company_research_models import BusinessBlock

_BANNED_PATTERN = re.compile(
    r"\b("
    r"debt[\s/]*equity|debt-to-equity|leverage ratio|current ratio|"
    r"net margin|gross margin|profit margin|operating margin|"
    r"p/?e ratio|price[\s-]*to[\s-]*earnings|price target|analyst rating|"
    r"consensus rating|mean target|upside to target|valuation multiple|"
    r"free cash flow|fcf\b|return on equity|roe\b|"
    r"trades at \d|/\d+x|\d+\.\d+x multiple"
    r")\b",
    re.IGNORECASE,
)

_BANNED_PHRASES = (
    "investors should monitor",
    "the company faces challenges",
    "the future depends on execution",
    "at a glance",
    "in conclusion",
    "overall, the company",
    "disciplined execution",
    "strategic positioning",
    "operational momentum",
    "strong operational momentum",
    "enhancing capabilities",
    "robust ecosystem",
    "robust execution",
    "scaling efficiently",
    "ability to scale efficiently",
    "scale efficiently",
    "strong market position",
    "well positioned",
    "best-in-class",
    "industry-leading",
    "commitment to innovation",
    "focus on innovation",
    "drive growth",
    "continue to grow",
    "favorable macro",
    "headwinds and tailwinds",
    "stability of demand",
    "stable demand",
)

_VAGUE_ONLY_RE = re.compile(
    r"^(?:strong|robust|solid|leading|innovative|strategic|dynamic|"
    r"competitive|scalable|efficient)\s+(?:business|model|position|growth|"
    r"operations|performance|momentum|ecosystem|capabilities)\.?$",
    re.IGNORECASE,
)

_GROWTH_IN_MECHANISM_RE = re.compile(
    r"\b(growth outlook|market expansion|increasing demand|adoption trends|"
    r"geographic expansion|upside potential)\b",
    re.IGNORECASE,
)

_ABILITY_WITHOUT_LIMIT_RE = re.compile(
    r"\bability to\b",
    re.IGNORECASE,
)

_MECHANISM_MARKERS = re.compile(
    r"\b(limited by|constrained by|depends on|requires|because|via|through|"
    r"when|if|until|bottleneck|supply|capacity|utilization|bundle|undercut|"
    r"discount|integrat|vertically|subsidi|metered|recognized|deployed)\b",
    re.IGNORECASE,
)

_VAGUE_DEPLOYMENT_RE = re.compile(
    r"\blarge[- ]scale deployments?\b",
    re.IGNORECASE,
)

_GENERIC_CHALLENGE_RE = re.compile(
    r"^(?:intense |strong )?(?:competition|competitive pressures?)\.?$",
    re.IGNORECASE,
)


class BusinessIntelligenceValidationError(ValueError):
    pass


def normalize_business_intelligence(
    block: BusinessBlock,
    *,
    fallback_industry: str | None = None,
) -> BusinessBlock:
    if fallback_industry and not block.industry.strip():
        block = block.model_copy(update={"industry": fallback_industry.strip()})

    return block.model_copy(
        update={
            "industry": block.industry.strip(),
            "primary_product": _clean_line(block.primary_product),
            "revenue_model": _clean_line(block.revenue_model),
            "primary_customers": _filter_bullets(block.primary_customers, 4),
            "how_they_make_money": _filter_bullets(
                block.how_they_make_money,
                3,
                extra_check=_reject_growth_commentary,
            ),
            "revenue_visibility": _filter_bullets(
                block.revenue_visibility,
                3,
                extra_check=_reject_capability_style,
            ),
            "advantages": _filter_bullets(block.advantages, 4),
            "challenges": _filter_bullets(
                block.challenges,
                4,
                extra_check=_reject_weak_challenge,
            ),
            "revenue_drivers": _filter_bullets(
                block.revenue_drivers,
                4,
                extra_check=_reject_non_revenue_driver,
            ),
            "constraints": _filter_bullets(
                block.constraints,
                4,
                extra_check=_reject_capability_style,
            ),
            "business_risks": _filter_bullets(block.business_risks, 4),
            "dependencies": _filter_bullets(block.dependencies, 4),
        }
    )


def _reject_growth_commentary(line: str) -> bool:
    return bool(_GROWTH_IN_MECHANISM_RE.search(line))


def _reject_capability_style(line: str) -> bool:
    if _ABILITY_WITHOUT_LIMIT_RE.search(line) and not _MECHANISM_MARKERS.search(line):
        return True
    if _VAGUE_DEPLOYMENT_RE.search(line) and not _MECHANISM_MARKERS.search(line):
        return True
    return False


def _reject_weak_challenge(line: str) -> bool:
    if _GENERIC_CHALLENGE_RE.match(line.strip()):
        return True
    lowered = line.lower()
    if "competition" in lowered and not _MECHANISM_MARKERS.search(line):
        return True
    return False


def _reject_non_revenue_driver(line: str) -> bool:
    lowered = line.lower()
    non_revenue = (
        "brand awareness",
        "employee morale",
        "corporate culture",
        "esg",
        "sustainability initiatives",
    )
    return any(phrase in lowered for phrase in non_revenue)


def _filter_bullets(
    items: list[str],
    limit: int,
    *,
    extra_check: Callable[[str], bool] | None = None,
) -> list[str]:
    cleaned: list[str] = []
    for raw in items:
        line = _clean_line(raw)
        if not line:
            continue
        if extra_check and extra_check(line):
            continue
        if _is_disallowed(line):
            continue
        cleaned.append(line)
    return _normalize_bullets(cleaned, limit)


def _is_disallowed(text: str) -> bool:
    if len(text) > 150:
        return True
    if _BANNED_PATTERN.search(text):
        return True
    if _VAGUE_ONLY_RE.match(text.strip()):
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BANNED_PHRASES)


def _normalize_bullets(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        line = _clean_line(raw)
        if not line:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= limit:
            break
    return out


def _clean_line(text: str) -> str:
    return " ".join(text.split()).strip()
