import json
from datetime import date
from pathlib import Path

import pytest

from app.broker.option_chain_table import build_option_chain_table, format_held_option_contracts_markdown
from app.broker.option_greeks import (
    estimate_delta_black_scholes,
    normalize_iv_percent,
    resolve_option_greeks,
    sanitize_delta,
)
from app.models.schwab_models import Instrument, Position
from app.models.schwab_option_chain_models import OptionChain, OptionContract
from tests.test_position_prompt_metrics import _make_position

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def test_sanitize_delta_rejects_schwab_placeholder():
    assert sanitize_delta(-999) is None
    assert sanitize_delta(0.48) == 0.48


def test_normalize_iv_percent_accepts_decimal_and_percent():
    assert normalize_iv_percent(-999) is None
    assert normalize_iv_percent(0.285) == pytest.approx(28.5)
    assert normalize_iv_percent(24.15) == 24.15


def test_resolve_option_greeks_estimates_delta_when_broker_placeholder():
    contract = OptionContract(
        putCall="CALL",
        symbol="AMZN  260620C00200000",
        strikePrice=200.0,
        expirationDate="2026-06-20T20:00:00.000+00:00",
        daysToExpiration=30,
        delta=-999,
        volatility=-999,
        markPrice=5.3,
        bidPrice=5.2,
        askPrice=5.4,
    )
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))

    greeks = resolve_option_greeks(
        contract,
        chain=chain,
        underlying_price=200.12,
        underlying_iv_percent=0.285,
        put_call="CALL",
        strike=200.0,
        expiration=date(2026, 6, 20),
    )

    assert greeks.delta is not None
    assert 0.3 <= greeks.delta <= 0.7
    assert greeks.iv_percent == 28.5
    assert "estimated" in greeks.delta_source


def test_build_option_chain_table_omits_placeholder_greeks():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    contract = chain.callExpDateMap["2026-06-20:30"]["200.0"][0]
    contract.delta = -999
    contract.volatility = -999

    table = build_option_chain_table(chain, strike_count=1, underlying_iv_percent=0.285)
    row = next(row for row in table.rows if row.strike == 200.0)

    assert row.call is not None
    assert row.call.delta is not None
    assert row.call.iv == 28.5


def test_format_held_option_contracts_includes_scenarios():
    chain = OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    position = _make_position(symbol="AMZN", long_qty=1, avg=12.2, market_value=1125.0, pnl=-95.0)
    position.instrument = Instrument(
        assetType="OPTION",
        cusip="037833100",
        symbol="AMZN  260620C00200000",
        underlyingSymbol="AMZN",
        strikePrice=200.0,
        expirationDate="2026-06-20T20:00:00.000+00:00",
    )

    markdown = format_held_option_contracts_markdown(
        chain=chain,
        positions=[position],
        symbol="AMZN",
        underlying_iv_percent=0.285,
    )

    assert "Profit scenarios" in markdown
    assert "delta -999" not in markdown
    assert "estimated" in markdown.lower() or "Black-Scholes" in markdown


def test_estimate_delta_black_scholes_atm_call():
    delta = estimate_delta_black_scholes(
        underlying=200.0,
        strike=200.0,
        days_to_expiration=30,
        put_call="CALL",
        iv_percent=28.0,
    )
    assert delta is not None
    assert 0.45 <= delta <= 0.60
