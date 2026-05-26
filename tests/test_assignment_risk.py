from datetime import date, timedelta

from app.broker.option_utils import (
    assignment_risk_level,
    classify_moneyness,
    days_to_expiration,
    format_assignment_risk_markdown,
    parse_expiration_from_option_symbol,
    position_expiration_date,
    summarize_assignment_risk,
)
from app.core.prompts import (
    AnalysisAction,
    SymbolContext,
    _build_action_prompt,
    build_symbol_prompt,
)
from app.models.schwab_models import Instrument, Position
from tests.test_option_utils import _make_option_position
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_parse_expiration_from_compact_symbol():
    assert parse_expiration_from_option_symbol("AAPL_041726P170") == date(2026, 4, 17)


def test_parse_expiration_from_occ_symbol():
    assert parse_expiration_from_option_symbol("AAPL  240315C00190000") == date(2024, 3, 15)


def test_position_expiration_date_prefers_instrument_field():
    position = _make_option_position(
        symbol="AAPL_041726P170",
        strike_price=170.0,
    )
    position.instrument.expirationDate = "2026-04-18"

    assert position_expiration_date(position) == date(2026, 4, 18)


def test_classify_moneyness_for_puts_and_calls():
    assert classify_moneyness(put_call="PUT", strike=170.0, underlying_price=165.0) == "ITM"
    assert classify_moneyness(put_call="PUT", strike=170.0, underlying_price=180.0) == "OTM"
    assert classify_moneyness(put_call="CALL", strike=190.0, underlying_price=195.0) == "ITM"


def test_assignment_risk_level_prioritizes_itm_near_expiration():
    assert assignment_risk_level(moneyness="ITM", days_to_expiry=1) == "critical"
    assert assignment_risk_level(moneyness="ITM", days_to_expiry=5) == "high"
    assert assignment_risk_level(moneyness="OTM", days_to_expiry=10) == "low"


def test_summarize_assignment_risk_flags_itm_csp():
    today = date(2026, 4, 10)
    expiration = today + timedelta(days=3)
    position = _make_option_position(
        symbol="AAPL_041726P170",
        put_call="PUT",
        short_qty=1,
        strike_price=170.0,
    )
    position.instrument.expirationDate = expiration.isoformat()
    position.optionStrategy = "cash_secured_put"

    summary = summarize_assignment_risk(
        [position],
        {"AAPL": 165.0},
        symbol="AAPL",
        within_days=14,
        as_of=today,
    )
    entries = summary["positions"]
    assert len(entries) == 1
    assert entries[0]["riskLevel"] == "high"
    assert entries[0]["moneyness"] == "ITM"
    assert entries[0]["assignmentCashRequired"] == 17_000.0


def test_format_assignment_risk_markdown_includes_scan_header():
    today = date(2026, 4, 10)
    position = _make_option_position(
        put_call="CALL",
        short_qty=1,
        strike_price=190.0,
    )
    position.instrument.expirationDate = (today + timedelta(days=2)).isoformat()
    position.optionStrategy = "covered_call"

    summary = summarize_assignment_risk(
        [position],
        {"AAPL": 195.0},
        symbol="AAPL",
        within_days=14,
        as_of=today,
    )
    markdown = format_assignment_risk_markdown(summary)

    assert "Assignment risk scan" in markdown
    assert "covered_call" in markdown
    assert "critical" in markdown


def test_build_action_prompt_for_assignment_risk():
    prompt = _build_action_prompt(
        AnalysisAction.ASSIGNMENT_RISK,
        "AAPL",
        None,
    )
    assert "assignment and call-away risk" in prompt.lower()
    assert "cash-secured puts" in prompt.lower()


def test_natural_assignment_risk_prompt_avoids_report_template():
    prompt = _build_action_prompt(
        AnalysisAction.ASSIGNMENT_RISK,
        "the portfolio",
        None,
        natural_delivery=True,
    )
    assert "assignment" in prompt.lower()
    assert "precomputed assignment risk scan" in prompt.lower()
    assert "Cover these points" not in prompt
    assert "1. **Expiring short options**" not in prompt
    assert "open with what you'd do first" in prompt.lower()


def test_build_symbol_prompt_includes_assignment_risk_block():
    account = _make_account()
    positions = [_make_position(symbol="AAPL")]
    ctx = SymbolContext(
        account=account,
        positions=positions,
        symbol="AAPL",
        action=AnalysisAction.ASSIGNMENT_RISK,
        assignment_risk_block="SYMBOL | UNDERLYING | STRATEGY",
        market_snapshot="snapshot",
        market_context="macro",
        option_chain="chain",
        research_context="research",
    )

    prompt = build_symbol_prompt(ctx=ctx)

    assert "ASSIGNMENT RISK SCAN (PRECOMPUTED)" in prompt
    assert "SYMBOL | UNDERLYING | STRATEGY" in prompt
    assert "EXPIRATION | DTE" in prompt
