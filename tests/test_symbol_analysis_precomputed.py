import json
from datetime import date, timedelta
from pathlib import Path

from app.models.analysis_models import (
    PortfolioAnalysisV1LLMResponse,
    SymbolAnalysisV1Envelope,
)
from app.models.intelligence_models import (
    OptionRollSuggestion,
    OptionsScorecard,
    SymbolIntelligence,
)
from app.models.schwab_models import Instrument, Position
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from app.services.symbol_analysis_precomputed_service import (
    SymbolAnalysisPrecomputedService,
)
from tests.test_position_prompt_metrics import _make_account, _make_position

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def _short_nvda_put_position() -> Position:
    near_exp = (date.today() + timedelta(days=3)).isoformat()
    return Position(
        shortQuantity=1.0,
        averagePrice=2.0,
        averageShortPrice=2.0,
        shortOpenProfitLoss=-730.0,
        currentDayProfitLoss=0.0,
        currentDayProfitLossPercentage=0.0,
        longQuantity=0.0,
        settledLongQuantity=0.0,
        settledShortQuantity=1.0,
        instrument=Instrument(
            assetType="OPTION",
            cusip="",
            symbol="NVDA  260529P00212500",
            putCall="PUT",
            underlyingSymbol="NVDA",
            strikePrice=212.5,
            expirationDate=near_exp,
        ),
        marketValue=-200.0,
        maintenanceRequirement=0.0,
        currentDayCost=0.0,
    )


def test_symbol_analysis_precomputed_builds_roll_close_hold_paths():
    near_exp = (date.today() + timedelta(days=3)).isoformat()
    roll_exp = (date.today() + timedelta(days=10)).isoformat()
    chain = OptionChain(
        symbol="NVDA",
        underlyingPrice=220.0,
        putExpDateMap={
            f"{near_exp}:3": {
                "212.5": [
                    OptionContract(
                        putCall="PUT",
                        symbol="NVDA",
                        strikePrice=212.5,
                        expirationDate=near_exp,
                        daysToExpiration=3,
                        delta=-0.44,
                        openInterest=800,
                        bid=1.2,
                        ask=1.35,
                    )
                ]
            },
            f"{roll_exp}:10": {
                "205.0": [
                    OptionContract(
                        putCall="PUT",
                        symbol="NVDA",
                        strikePrice=205.0,
                        expirationDate=roll_exp,
                        daysToExpiration=10,
                        delta=-0.28,
                        openInterest=1200,
                        bid=2.5,
                        ask=2.7,
                    )
                ]
            },
        },
    )
    intelligence = SymbolIntelligence(
        symbol="NVDA",
        roll_suggestions=[
            OptionRollSuggestion(
                side="put",
                current_strike=212.5,
                current_expiration=near_exp,
                suggested_strike=205.0,
                suggested_expiration=roll_exp,
                current_delta=-0.44,
                suggested_delta=-0.28,
                estimated_credit=1.15,
                rationale="roll",
            )
        ],
        options_scorecard=OptionsScorecard(underlying_price=220.0),
    )
    account = _make_account(liquidation_value=100_000)
    positions = [_short_nvda_put_position(), _make_position(symbol="NVDA", market_value=800)]

    precomputed = SymbolAnalysisPrecomputedService.build(
        symbol="NVDA",
        account=account,
        positions=positions,
        intelligence=intelligence,
        option_chain=chain,
        underlying_price=220.0,
    )

    assert precomputed is not None
    assert len(precomputed.held_option_outcomes) == 1
    outcome = precomputed.held_option_outcomes[0]
    assert outcome.drivers.open_pnl_pct is not None
    assert outcome.drivers.action_trigger is not None
    assert outcome.close.cost_per_contract == 135.0
    assert outcome.roll is not None
    assert outcome.roll.open_leg.strike == 205.0
    assert outcome.roll.net_credit_per_contract == 115.0
    assert outcome.hold.in_the_money is False
    assert len(outcome.compare_paths) >= 2
    roll_card = next(c for c in outcome.compare_paths if c.path == "roll")
    assert any("Net credit" in line for line in roll_card.lines)


def test_symbol_analysis_v1_envelope_serializes_camel_case():
    envelope = SymbolAnalysisV1Envelope(
        analysis=PortfolioAnalysisV1LLMResponse(
            summary="Roll the short put.",
            recommendedAction={
                "title": "Roll the option",
                "reason": "Loss and high delta.",
                "symbol": "NVDA",
            },
            sections=[
                {
                    "title": "Outcome comparison",
                    "bullets": ["Net credit about $115 per contract."],
                }
            ],
        ),
        precomputed=SymbolAnalysisPrecomputedService.build(
            symbol="NVDA",
            account=_make_account(),
            positions=[_short_nvda_put_position()],
            intelligence=SymbolIntelligence(symbol="NVDA"),
            option_chain=None,
            underlying_price=220.0,
        ),
    )

    payload = json.loads(envelope.model_dump_json(by_alias=True))
    assert "analysis" in payload
    assert payload["analysis"]["recommendedAction"]["symbol"] == "NVDA"
    assert payload["precomputed"] is not None
    assert payload["precomputed"]["heldOptionOutcomes"]


def test_build_symbol_prompt_includes_precomputed_outcomes_when_json():
    from app.core.prompts import SymbolContext, build_symbol_prompt
    from app.models.symbol_analysis_precomputed_models import SymbolAnalysisPrecomputed

    ctx = SymbolContext(
        symbol="NVDA",
        account=_make_account(),
        positions=[_short_nvda_put_position()],
        precomputed=SymbolAnalysisPrecomputed(symbol="NVDA", underlying_price=220.0),
    )

    prompt = build_symbol_prompt(ctx, json_response=True)

    assert "PRECOMPUTED OUTCOMES" in prompt
    assert '"underlyingPrice": 220.0' in prompt or '"underlyingPrice":220.0' in prompt
