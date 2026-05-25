import json
from datetime import date
from pathlib import Path

from app.broker.option_chain_table import (
    build_option_chain_tables_for_positions,
    format_held_option_contracts_markdown,
    lookup_option_contract,
)
from app.models.schwab_models import Instrument, Position
from app.models.schwab_option_chain_models import OptionChain
from app.services.chat_service import ChatService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.core.prompts import AnalysisAction
from tests.test_position_prompt_metrics import _make_position

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def _make_amzn_call_position() -> Position:
    position = _make_position(
        symbol="AMZN",
        long_qty=1,
        avg=12.2,
        market_value=1125.0,
        pnl=-95.0,
    )
    position.instrument = Instrument(
        assetType="OPTION",
        cusip="037833100",
        symbol="AMZN  260620C00200000",
        underlyingSymbol="AMZN",
        strikePrice=200.0,
        expirationDate="2026-06-20T20:00:00.000+00:00",
    )
    return position


def test_lookup_option_contract_returns_held_strike_greeks():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    contract = lookup_option_contract(
        chain,
        expiration=date(2026, 6, 20),
        strike=200.0,
        put_call="CALL",
    )

    assert contract is not None
    assert contract.delta == 0.48


def test_format_held_option_contracts_includes_delta_iv_and_underlying():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    chain.symbol = "AMZN"
    position = _make_amzn_call_position()

    markdown = format_held_option_contracts_markdown(
        chain=chain,
        positions=[position],
        symbol="AMZN",
    )

    assert "Held option contracts" in markdown
    assert "delta" in markdown
    assert "IV" in markdown
    assert "underlying" in markdown


def test_resolve_option_chain_block_includes_held_contracts_for_free_form():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    chain.symbol = "AMZN"
    position = _make_amzn_call_position()
    service = PromptEnrichmentService()

    block = service.resolve_option_chain_block(
        chain=chain,
        action=AnalysisAction.FREE_FORM,
        positions=[position],
        symbol="AMZN",
        strike_count=2,
    )

    assert "Held option contracts" in block
    assert "Option chain near held expiration" in block or "Option chain" in block


def test_build_option_chain_tables_for_positions_uses_held_expiration():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    position = _make_amzn_call_position()

    tables = build_option_chain_tables_for_positions(
        chain,
        [position],
        "AMZN",
        strike_count=1,
    )

    assert tables
    assert tables[0].expiration == "2026-06-20"


def test_substantive_free_form_question_includes_portfolio_context():
    assert ChatService.should_include_portfolio_context(
        is_first_chat=False,
        action=AnalysisAction.FREE_FORM,
        recent_messages=[{"role": "assistant", "content": "Hold the call."}],
        user_prompt="What are the odds my call gains another 30%?",
    )
