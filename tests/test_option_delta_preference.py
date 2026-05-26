from app.broker.option_delta_preference import (
    assignment_delta_threshold,
    default_delta_band_for_risk,
    resolve_option_delta_band,
)
from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelStrategyConfig,
)


def test_default_delta_bands_by_risk():
    conservative = default_delta_band_for_risk("conservative")
    moderate = default_delta_band_for_risk("moderate")
    aggressive = default_delta_band_for_risk("aggressive")

    assert conservative.min_delta == 0.10
    assert conservative.max_delta == 0.15
    assert conservative.profile_label == "conservative"

    assert moderate.min_delta == 0.20
    assert moderate.max_delta == 0.30
    assert moderate.profile_label == "balanced"

    assert aggressive.min_delta == 0.35
    assert aggressive.max_delta == 0.50
    assert aggressive.profile_label == "aggressive"


def test_resolve_option_delta_band_uses_wheel_config():
    profile = UserInvestmentProfile(
        userId="user-1",
        primaryStrategy=InvestmentStrategy.WHEEL,
        riskTolerance="moderate",
        wheel=WheelStrategyConfig(
            targetDeltaMin=0.35,
            targetDeltaMax=0.50,
        ),
    )

    band = resolve_option_delta_band(profile)

    assert band.min_delta == 0.35
    assert band.max_delta == 0.50
    assert band.profile_label == "aggressive"


def test_resolve_option_delta_band_falls_back_to_risk_without_wheel():
    profile = UserInvestmentProfile(
        userId="user-1",
        primaryStrategy=InvestmentStrategy.DIVIDEND,
        riskTolerance="conservative",
    )

    band = resolve_option_delta_band(profile)

    assert band.min_delta == 0.10
    assert band.max_delta == 0.15


def test_resolve_option_strategy_preferences_uses_wheel_dte():
    from app.broker.option_delta_preference import resolve_option_strategy_preferences

    profile = UserInvestmentProfile(
        userId="user-1",
        primaryStrategy=InvestmentStrategy.WHEEL,
        riskTolerance="moderate",
        wheel=WheelStrategyConfig(preferredDteDays=14),
    )

    prefs = resolve_option_strategy_preferences(profile)

    assert prefs.preferred_dte_days == 14
    assert prefs.delta_band.min_delta == 0.20


def test_assignment_threshold_scales_with_band():
    conservative = default_delta_band_for_risk("conservative")
    moderate = default_delta_band_for_risk("moderate")
    aggressive = default_delta_band_for_risk("aggressive")

    assert assignment_delta_threshold(conservative) == 0.20
    assert assignment_delta_threshold(moderate) == 0.40
    assert assignment_delta_threshold(aggressive) == 0.45
