import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.api.get_option_chain_debug_route import (
    _summarize_chain,
    get_option_chain_debug,
)
from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.models.schwab_option_chain_models import OptionChain

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def test_schwab_market_builder_logs_validation_errors(caplog):
    adapter = MagicMock()
    adapter.get_option_chains.return_value = {
        "symbol": "AAPL",
        "callExpDateMap": {
            "2026-06-20:30": {
                "200.0": [{"putCall": "CALL"}],
            }
        },
    }
    builder = SchwabMarketBuilder(adapter)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValidationError):
            builder.get_option_chains(access_token="token", symbol="AAPL")

    assert any(
        "Option chain validation failed for AAPL" in record.message
        for record in caplog.records
    )


def test_summarize_chain_reports_nearest_expiration_strikes():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    summary = _summarize_chain(chain, strike_count=2)

    assert summary["nearestExpiration"] == "2026-06-20:30"
    assert 200.0 in summary["nearestExpirationCallStrikes"]
    assert 190.0 in summary["nearestExpirationPutStrikes"]


def test_get_option_chain_debug_returns_parsed_summary():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))

    market_service = MagicMock()
    market_service.get_option_chains.return_value = chain

    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.side_effect = RuntimeError("skip positions")

    schwab_auth_service = MagicMock()
    token = MagicMock()
    token.access_token = "schwab-token"
    schwab_auth_service.get_valid_token_by_user_id.return_value = token

    payload = get_option_chain_debug(
        symbol="AAPL",
        strike_count=5,
        include_raw_chain=False,
        user_id="user-1",
        market_service=market_service,
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
    )

    assert payload["symbol"] == "AAPL"
    assert payload["summary"]["underlyingPrice"] == 200.12
    assert payload["scorecard"] is not None
    assert "Underlying: AAPL @ $200.12" in payload["markdownPreview"]
    assert "Expiration: 2026-06-20 (30 DTE)" in payload["markdownPreview"]
    assert "Call Delta" in payload["markdownPreview"]
    assert "rawChain" not in payload
    market_service.get_option_chains.assert_called_once()


def test_get_option_chain_debug_can_include_raw_chain():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))

    market_service = MagicMock()
    market_service.get_option_chains.return_value = chain

    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.side_effect = RuntimeError("skip positions")

    schwab_auth_service = MagicMock()
    token = MagicMock()
    token.access_token = "schwab-token"
    schwab_auth_service.get_valid_token_by_user_id.return_value = token

    payload = get_option_chain_debug(
        symbol="AAPL",
        strike_count=5,
        include_raw_chain=True,
        user_id="user-1",
        market_service=market_service,
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
    )

    assert payload["rawChain"]["symbol"] == "AAPL"
    assert "callExpDateMap" in payload["rawChain"]
