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
    "enhancing capabilities",
    "robust ecosystem",
    "scaling efficiently",
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
            "revenue_visibility": _filter_bullets(block.revenue_visibility, 2),
            "advantages": _filter_bullets(block.advantages, 5),
            "challenges": _filter_bullets(block.challenges, 5),
            "growth_drivers": _filter_bullets(
                block.growth_drivers,
                5,
                extra_check=_reject_non_revenue_growth,
            ),
            "business_risks": _filter_bullets(block.business_risks, 5),
            "dependencies": _filter_bullets(block.dependencies, 5),
        }
    )


def _reject_growth_commentary(line: str) -> bool:
    return bool(_GROWTH_IN_MECHANISM_RE.search(line))


def _reject_non_revenue_growth(line: str) -> bool:
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
    if len(text) > 160:
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
