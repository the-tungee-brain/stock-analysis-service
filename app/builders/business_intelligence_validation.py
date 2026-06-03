from __future__ import annotations

import re

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
            "business_model": _clean_line(block.business_model),
            "primary_customers": _filter_bullets(block.primary_customers, 5),
            "how_they_make_money": _filter_bullets(block.how_they_make_money, 3),
            "advantages": _filter_bullets(block.advantages, 5),
            "challenges": _filter_bullets(block.challenges, 5),
            "growth_drivers": _filter_bullets(block.growth_drivers, 5),
            "business_risks": _filter_bullets(block.business_risks, 5),
            "dependencies": _filter_bullets(block.dependencies, 5),
        }
    )


def _filter_bullets(items: list[str], limit: int) -> list[str]:
    cleaned: list[str] = []
    for raw in items:
        line = _clean_line(raw)
        if not line:
            continue
        if _is_disallowed(line):
            continue
        cleaned.append(line)
    return _normalize_bullets(cleaned, limit)


def _is_disallowed(text: str) -> bool:
    if len(text) > 220:
        return True
    if _BANNED_PATTERN.search(text):
        return True
    lowered = text.lower()
    return any(phrase in lowered for phrase in _BANNED_PHRASES)


def _all_text_fields(block: BusinessBlock) -> list[str]:
    return [
        block.industry,
        block.primary_product,
        block.revenue_model,
        block.business_model,
        *block.primary_customers,
        *block.how_they_make_money,
        *block.advantages,
        *block.challenges,
        *block.growth_drivers,
        *block.business_risks,
        *block.dependencies,
    ]


def _assert_no_financial_overlap(text: str) -> None:
    if not text.strip():
        return
    if _BANNED_PATTERN.search(text):
        raise BusinessIntelligenceValidationError(
            f"Business content must not include financial/valuation metrics: {text!r}",
        )


def _assert_not_banned_prose(text: str) -> None:
    lowered = text.lower()
    if len(text) > 220:
        raise BusinessIntelligenceValidationError(
            "Business bullets must be short — move detail into separate fields.",
        )
    if any(phrase in lowered for phrase in _BANNED_PHRASES):
        raise BusinessIntelligenceValidationError(
            f"Generic AI phrasing detected: {text!r}",
        )


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
