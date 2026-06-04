from __future__ import annotations

import re

from app.models.position_guidance_models import PositionGuidanceItem, PositionVerdict

class PassiveExplanationViolation(ValueError):
    pass


# Advisory / interpretive language — not allowed in passive explanation layers.
_BANNED_PHRASES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bshould\b",
        r"\bwe would\b",
        r"\bi would\b",
        r"\bbetter outlook\b",
        r"\bworse outlook\b",
        r"\bweakening\b",
        r"\bimproving\b",
        r"\bpressure\b",
        r"\brecommend\b",
        r"\bsuggest\b",
        r"\bconsider (?:closing|trimming|selling|buying)\b",
    )
)

# Trade verbs as advice (not when part of verdict: line).
_BANNED_ADVICE_VERBS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:you |i )?should (?:buy|sell|close|trim|hold)\b",
        r"\b(?:buy|sell) (?:more|shares|contracts)\b",
        r"\bclose (?:the|your|this)\b",
        r"\btrim (?:the|your|this)\b",
        r"\bhold (?:the|your|this) (?:position|leg)\b",
    )
)

_VERDICT_LINE = re.compile(
    r"^\s*verdict:\s*(HOLD|TRIM|REVIEW_SELL|EXIT|REVIEW_CLOSE|CLOSE|ROLL|REVIEW_ASSIGNMENT_RISK)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_verdict_lines(text: str) -> str:
    return _VERDICT_LINE.sub("", text)


def validate_passive_explanation_text(text: str) -> None:
    """Fail if text reads like advice rather than a scoring trace."""
    if not text or not text.strip():
        return
    check = _strip_verdict_lines(text)
    for pattern in _BANNED_PHRASES + _BANNED_ADVICE_VERBS:
        if pattern.search(check):
            raise PassiveExplanationViolation(
                f"Passive explanation contains banned advisory language: {pattern.pattern}"
            )


def validate_verdict_in_trace(
    *,
    text: str,
    expected_verdict: PositionVerdict,
    leg_label: str | None = None,
) -> None:
    validate_passive_explanation_text(text)
    matches = _VERDICT_LINE.findall(text)
    if not matches:
        return
    for found in matches:
        if found.upper() != expected_verdict.upper():
            label = leg_label or "leg"
            raise PassiveExplanationViolation(
                f"{label}: trace verdict {found} != engine verdict {expected_verdict}"
            )


def validate_trace_matches_guidance(
    trace: str,
    items: list[PositionGuidanceItem],
) -> None:
    """Structural checks for engine-built trace (not advisory-language checks)."""
    for item in items:
        if item.verdict not in trace:
            raise PassiveExplanationViolation(
                f"Trace missing engine verdict {item.verdict} for {item.display_label}"
            )
        if item.primary_reason not in trace:
            raise PassiveExplanationViolation(
                f"Trace missing primary_reason for {item.display_label}"
            )
        for c in item.scoring_contributors:
            if c.label not in trace:
                raise PassiveExplanationViolation(
                    f"Trace missing contributor label for {item.display_label}: {c.label}"
                )


def validate_no_unknown_drivers(
    text: str,
    allowed_driver_codes: set[str],
    allowed_labels: set[str],
) -> None:
    """Reject driver names in text that are not from engine contributors."""
    validate_passive_explanation_text(text)
    for pattern in (
        r"\b([A-Z][A-Z0-9_]{2,})\b",
        r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b",
    ):
        for match in re.finditer(pattern, text):
            token = match.group(1).strip()
            if token in {
                "HOLD",
                "TRIM",
                "EXIT",
                "CLOSE",
                "ROLL",
                "REVIEW",
                "SELL",
                "CALL",
                "PUT",
                "TRUE",
                "FALSE",
            }:
                continue
            normalized = token.upper().replace(" ", "_")
            if (
                normalized not in allowed_driver_codes
                and token not in allowed_labels
                and len(token) > 3
            ):
                # Only strict-fail obvious invented driver codes
                if "_" in normalized and normalized.isupper():
                    raise PassiveExplanationViolation(
                        f"Unknown driver token in explanation: {token}"
                    )
