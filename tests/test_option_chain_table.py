import json
from pathlib import Path

from app.broker.option_chain_table import build_option_chain_table
from app.models.schwab_option_chain_models import OptionChain
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
    _build_option_chain_preview,
)
from tests.test_position_prompt_metrics import _make_account, _make_position
from tests.test_intelligence_layer import _research_context

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def test_build_option_chain_table_returns_up_down_rows():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    table = build_option_chain_table(chain, strike_count=1)

    assert table is not None
    assert table.expiration == "2026-06-20"
    assert table.underlying_price == 200.12
    assert [row.strike for row in table.rows] == [195.0, 200.0]
    assert table.rows[0].put is not None
    assert table.rows[0].put.bid == 4.1
    assert table.rows[1].call is not None
    assert table.rows[1].call.ask == 5.4
    assert table.rows[1].call.iv == 23.8


def test_build_option_chain_preview_serializes_for_symbol_intelligence():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    preview = _build_option_chain_preview(chain)

    assert preview is not None
    payload = preview.model_dump(mode="json", by_alias=True)
    assert payload["underlyingPrice"] == 200.12
    assert payload["strikeCount"] == 5
    row_195 = next(row for row in payload["rows"] if row["strike"] == 195.0)
    assert row_195["call"]["bid"] == 8.5
    row_190 = next(row for row in payload["rows"] if row["strike"] == 190.0)
    assert row_190["put"]["delta"] == -0.22


def test_build_symbol_intelligence_includes_option_chain_preview():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    service = PortfolioIntelligenceService(
        peer_comparison_service=__import__("unittest.mock").mock.MagicMock(),
        enriched_news_service=__import__("unittest.mock").mock.MagicMock(),
    )

    intelligence = service.build_symbol_intelligence(
        research=_research_context(),
        positions=[_make_position(symbol="AAPL")],
        account=_make_account(),
        symbol="AAPL",
        option_chain=chain,
        include_peers=False,
    )

    assert intelligence.option_chain_preview is not None
    assert len(intelligence.option_chain_preview.rows) >= 2
