import json
from pathlib import Path

import pytest

from app.broker.option_chain_table import (
    build_option_chain_table,
    fair_option_price,
    quoted_last,
)
from app.models.schwab_option_chain_models import OptionChain, OptionContract
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
    assert table.days_to_expiration == 30
    assert table.symbol == "AAPL"
    assert table.underlying_price == 200.12
    assert [row.strike for row in table.rows] == [195.0, 200.0]
    assert table.rows[0].put is not None
    assert table.rows[0].put.bid == 4.1
    assert table.rows[0].put.mark == 4.2
    assert table.rows[1].call is not None
    assert table.rows[1].call.ask == 5.4
    assert table.rows[1].call.mark == 5.3
    assert table.rows[1].call.last_price == 5.3
    assert table.rows[1].call.theta == -0.11
    assert table.rows[1].call.iv == 23.8


def test_build_option_chain_preview_estimates_greeks_when_broker_sends_placeholders():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    contract = chain.callExpDateMap["2026-06-20:30"]["200.0"][0]
    contract.delta = -999
    contract.volatility = -999
    contract.theta = -999
    chain.volatility = -999

    preview = _build_option_chain_preview(chain, underlying_iv_percent=0.285)

    assert preview is not None
    row_200 = next(row for row in preview.rows if row.strike == 200.0)
    assert row_200.call is not None
    assert row_200.call.delta is not None
    assert 0.3 <= row_200.call.delta <= 0.7
    assert row_200.call.iv == pytest.approx(28.5)


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


def test_fair_option_price_uses_theoretical_when_live_quotes_missing():
    contract = OptionContract(
        putCall="CALL",
        symbol="AAPL  260526C00310000",
        strikePrice=310.0,
        expirationDate="2026-05-26T20:00:00.000+00:00",
        daysToExpiration=2,
        theoreticalOptionValue=1.21,
        delta=0.39,
    )

    assert fair_option_price(contract) == 1.21


def test_quoted_last_uses_prior_close_when_last_trade_missing():
    contract = OptionContract(
        putCall="CALL",
        symbol="AAPL  260526C00310000",
        strikePrice=310.0,
        expirationDate="2026-05-26T20:00:00.000+00:00",
        daysToExpiration=2,
        closePrice=1.21,
        delta=0.39,
    )

    assert quoted_last(contract) == 1.21
    assert fair_option_price(contract) == 1.21
