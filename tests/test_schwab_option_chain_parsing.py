import json
from pathlib import Path

from app.core.prompts import AnalysisAction
from app.models.schwab_option_chain_models import OptionChain
from app.services.intelligence.options_scoring_service import OptionsScoringService
from app.services.prompt_enrichment_service import PromptEnrichmentService


FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def test_option_chain_model_validates_schwab_fixture():
    raw = json.loads(FIXTURE.read_text())
    chain = OptionChain.model_validate(raw)

    assert chain.symbol == "AAPL"
    assert chain.underlyingPrice == 200.12
    assert chain.underlying is not None
    assert chain.underlying.last == 200.12

    exp_key = "2026-06-20:30"
    assert exp_key in chain.callExpDateMap
    assert exp_key in chain.putExpDateMap

    call = chain.callExpDateMap[exp_key]["200.0"][0]
    assert call.putCall == "CALL"
    assert call.strikePrice == 200.0
    assert call.delta == 0.48
    assert call.openInterest == 8900

    put = chain.putExpDateMap[exp_key]["190.0"][0]
    assert put.putCall == "PUT"
    assert put.strikePrice == 190.0
    assert put.delta == -0.22


def test_option_chain_markdown_includes_put_only_strikes():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    markdown = PromptEnrichmentService().build_option_chain_markdown(chain, strike_count=5)

    assert "190.00" in markdown
    assert "-0.22" in markdown


def test_option_chain_markdown_respects_up_down_strike_count():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    markdown = PromptEnrichmentService().build_option_chain_markdown(chain, strike_count=1)

    assert "195.00" in markdown
    assert "200.00" in markdown
    assert "190.00" not in markdown


def test_option_chain_markdown_uses_nearest_expiration_and_greeks():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    markdown = PromptEnrichmentService().build_option_chain_markdown(chain, strike_count=5)

    assert "Underlying: AAPL @ $200.12" in markdown
    assert "Expiration: 2026-06-20 (30 DTE)" in markdown
    assert "per share" in markdown
    assert "Call Mark" in markdown
    assert "200.00" in markdown
    assert "0.48" in markdown
    assert "8,900" in markdown
    assert "24%" in markdown or "24.0%" in markdown
    assert "8.60" in markdown or "8.6" in markdown


def test_resolve_option_chain_block_omits_table_for_tax_angle():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    service = PromptEnrichmentService()

    block = service.resolve_option_chain_block(
        chain,
        AnalysisAction.TAX_ANGLE,
        has_options_scorecard=True,
    )

    assert "Option chain table omitted" in block
    assert "Call Mark" not in block


def test_resolve_option_chain_block_uses_scorecard_for_daily_summary():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    service = PromptEnrichmentService()

    block = service.resolve_option_chain_block(
        chain,
        AnalysisAction.DAILY_SUMMARY,
        has_options_scorecard=True,
    )

    assert "Full strike table omitted" in block
    assert "options scorecard" in block


def test_resolve_option_chain_block_keeps_full_chain_for_assignment_risk():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    service = PromptEnrichmentService()

    block = service.resolve_option_chain_block(
        chain,
        AnalysisAction.ASSIGNMENT_RISK,
        has_options_scorecard=True,
    )

    assert "Underlying: AAPL" in block
    assert "Call Mark" in block


def test_options_scorecard_reads_parsed_chain():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    scorecard = OptionsScoringService.build_scorecard(chain)

    assert scorecard is not None
    assert scorecard.underlying_price == 200.12
    assert scorecard.covered_call_candidates
    assert scorecard.csp_candidates
    assert scorecard.covered_call_candidates[0].strike == 200.0
    assert scorecard.csp_candidates[0].strike == 190.0
