from datetime import date, timedelta

from app.models.schwab_option_chain_models import OptionChain, OptionContract
from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelStrategyConfig,
)
from app.services.intelligence.options_scoring_service import OptionsScoringService


def _put_contract(*, strike: float, delta: float, dte: int) -> OptionContract:
    expiration = (date.today() + timedelta(days=dte)).isoformat()
    return OptionContract(
        putCall="PUT",
        symbol="AAPL",
        strikePrice=strike,
        expirationDate=expiration,
        daysToExpiration=dte,
        delta=-abs(delta),
        openInterest=1200,
        bidPrice=2.5,
        askPrice=2.7,
    )


def _call_contract(*, strike: float, delta: float, dte: int) -> OptionContract:
    expiration = (date.today() + timedelta(days=dte)).isoformat()
    return OptionContract(
        putCall="CALL",
        symbol="AAPL",
        strikePrice=strike,
        expirationDate=expiration,
        daysToExpiration=dte,
        delta=abs(delta),
        openInterest=1200,
        bidPrice=2.5,
        askPrice=2.7,
    )


def test_scorecard_returns_at_most_three_candidates_per_side():
    exp = (date.today() + timedelta(days=7)).isoformat()
    chain = OptionChain(
        symbol="AAPL",
        underlyingPrice=200.0,
        putExpDateMap={
            f"{exp}:7": {
                f"{190 - i}.0": [_put_contract(strike=190 - i, delta=0.22 + i * 0.01, dte=7)]
                for i in range(6)
            }
        },
        callExpDateMap={
            f"{exp}:7": {
                f"{210 + i}.0": [_call_contract(strike=210 + i, delta=0.22 + i * 0.01, dte=7)]
                for i in range(5)
            }
        },
    )

    scorecard = OptionsScoringService.build_scorecard(chain)

    assert scorecard is not None
    assert len(scorecard.csp_candidates) <= 3
    assert len(scorecard.covered_call_candidates) <= 3


def test_scorecard_prefers_in_band_deltas_for_conservative_profile():
    exp = (date.today() + timedelta(days=7)).isoformat()
    profile = UserInvestmentProfile(
        userId="user-1",
        primaryStrategy=InvestmentStrategy.WHEEL,
        riskTolerance="conservative",
        wheel=WheelStrategyConfig(
            targetDeltaMin=0.10,
            targetDeltaMax=0.15,
            preferredDteDays=7,
        ),
    )
    chain = OptionChain(
        symbol="AAPL",
        underlyingPrice=200.0,
        putExpDateMap={
            f"{exp}:7": {
                "195.0": [_put_contract(strike=195.0, delta=0.11, dte=7)],
                "190.0": [_put_contract(strike=190.0, delta=0.14, dte=7)],
                "185.0": [_put_contract(strike=185.0, delta=0.12, dte=7)],
                "180.0": [_put_contract(strike=180.0, delta=0.13, dte=7)],
                "175.0": [_put_contract(strike=175.0, delta=0.25, dte=7)],
                "170.0": [_put_contract(strike=170.0, delta=0.40, dte=7)],
            }
        },
    )

    scorecard = OptionsScoringService.build_scorecard(chain, profile=profile)

    assert scorecard is not None
    assert len(scorecard.csp_candidates) == 3
    for candidate in scorecard.csp_candidates:
        assert 0.10 <= abs(candidate.delta) <= 0.15
    assert 175.0 not in {c.strike for c in scorecard.csp_candidates}
    assert 170.0 not in {c.strike for c in scorecard.csp_candidates}


def test_scorecard_prefers_preferred_dte_from_profile():
    near_exp = (date.today() + timedelta(days=7)).isoformat()
    far_exp = (date.today() + timedelta(days=21)).isoformat()
    profile = UserInvestmentProfile(
        userId="user-1",
        primaryStrategy=InvestmentStrategy.CSP_INCOME,
        riskTolerance="moderate",
        wheel=WheelStrategyConfig(
            targetDeltaMin=0.20,
            targetDeltaMax=0.30,
            preferredDteDays=7,
        ),
    )
    chain = OptionChain(
        symbol="NVDA",
        underlyingPrice=220.0,
        putExpDateMap={
            f"{near_exp}:7": {
                "200.0": [_put_contract(strike=200.0, delta=0.25, dte=7)],
            },
            f"{far_exp}:21": {
                "200.0": [_put_contract(strike=200.0, delta=0.25, dte=21)],
            },
        },
    )

    scorecard = OptionsScoringService.build_scorecard(chain, profile=profile)

    assert scorecard is not None
    assert scorecard.csp_candidates
    assert scorecard.csp_candidates[0].expiration[:10] == near_exp
