from __future__ import annotations

from dataclasses import dataclass

from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile

RiskTolerance = str


@dataclass(frozen=True)
class OptionDeltaBand:
    min_delta: float
    max_delta: float
    profile_label: str
    description: str


RISK_DELTA_BANDS: dict[RiskTolerance, OptionDeltaBand] = {
    "conservative": OptionDeltaBand(
        min_delta=0.10,
        max_delta=0.15,
        profile_label="conservative",
        description="lower assignment probability, lower premium",
    ),
    "moderate": OptionDeltaBand(
        min_delta=0.20,
        max_delta=0.30,
        profile_label="balanced",
        description="wheel sweet spot",
    ),
    "aggressive": OptionDeltaBand(
        min_delta=0.35,
        max_delta=0.50,
        profile_label="aggressive",
        description="higher income, higher assignment risk",
    ),
}

DEFAULT_DELTA_BAND = RISK_DELTA_BANDS["moderate"]


def default_delta_band_for_risk(risk_tolerance: RiskTolerance | None) -> OptionDeltaBand:
    return RISK_DELTA_BANDS.get(risk_tolerance or "moderate", DEFAULT_DELTA_BAND)


def resolve_option_delta_band(
    profile: UserInvestmentProfile | None,
) -> OptionDeltaBand:
    if profile is None:
        return DEFAULT_DELTA_BAND

    wheel = profile.wheel
    if wheel is not None and profile.primary_strategy in {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }:
        risk_band = default_delta_band_for_risk(profile.risk_tolerance)
        if (
            abs(wheel.target_delta_min - risk_band.min_delta) < 0.001
            and abs(wheel.target_delta_max - risk_band.max_delta) < 0.001
        ):
            return risk_band
        return OptionDeltaBand(
            min_delta=wheel.target_delta_min,
            max_delta=wheel.target_delta_max,
            profile_label=_infer_profile_label(
                wheel.target_delta_min, wheel.target_delta_max
            ),
            description=_describe_custom_range(
                wheel.target_delta_min, wheel.target_delta_max
            ),
        )

    return default_delta_band_for_risk(profile.risk_tolerance)


def assignment_delta_threshold(band: OptionDeltaBand) -> float:
    if band.profile_label == "conservative":
        return 0.20
    if band.profile_label == "aggressive":
        return 0.45
    return 0.40


def format_delta_band_summary(band: OptionDeltaBand) -> str:
    return (
        f"|delta| {band.min_delta:.2f}–{band.max_delta:.2f} "
        f"({band.profile_label}: {band.description})"
    )


def format_delta_band_prompt_line(band: OptionDeltaBand) -> str:
    return (
        f"- Target short-option |delta|: {band.min_delta:.2f}–{band.max_delta:.2f} "
        f"— {band.profile_label} ({band.description})"
    )


def _infer_profile_label(min_delta: float, max_delta: float) -> str:
    mid = (min_delta + max_delta) / 2.0
    if mid <= 0.17:
        return "conservative"
    if mid >= 0.32:
        return "aggressive"
    return "balanced"


def _describe_custom_range(min_delta: float, max_delta: float) -> str:
    label = _infer_profile_label(min_delta, max_delta)
    template = RISK_DELTA_BANDS.get(label)
    if template is not None:
        return template.description
    return "custom delta range"
