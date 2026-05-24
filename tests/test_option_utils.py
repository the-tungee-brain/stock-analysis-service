from app.broker.option_utils import (
    cash_secured_put_reserved_cash,
    parse_strike_from_option_symbol,
    summarize_csp_cash_reserves,
    total_csp_reserved_cash,
)
from app.core.prompts import _build_account_summary, _enrich_positions_table
from app.models.schwab_models import Instrument, Position
from tests.test_position_prompt_metrics import _make_account, _make_position


def _make_option_position(
    *,
    symbol: str = "AAPL_041726P170",
    put_call: str = "PUT",
    short_qty: float = 1,
    strike_price: float | None = 170.0,
) -> Position:
    return Position(
        shortQuantity=short_qty,
        averagePrice=2.5,
        currentDayProfitLoss=10.0,
        currentDayProfitLossPercentage=0.5,
        longQuantity=0,
        settledLongQuantity=0,
        settledShortQuantity=short_qty,
        instrument=Instrument(
            assetType="OPTION",
            cusip="",
            symbol=symbol,
            putCall=put_call,
            underlyingSymbol="AAPL",
            strikePrice=strike_price,
        ),
        marketValue=-250.0,
        maintenanceRequirement=0.0,
        averageShortPrice=2.5,
        shortOpenProfitLoss=50.0,
        currentDayCost=0.0,
    )


def test_parse_strike_from_occ_symbol():
    assert parse_strike_from_option_symbol("AAPL  240315C00190000") == 190.0


def test_parse_strike_from_compact_symbol():
    assert parse_strike_from_option_symbol("AAPL_041726C190") == 190.0
    assert parse_strike_from_option_symbol("AAPL_041726P170") == 170.0


def test_cash_secured_put_reserved_cash_uses_strike_times_100_shares():
    position = _make_option_position(short_qty=2, strike_price=170.0)
    assert cash_secured_put_reserved_cash(position) == 34_000.0


def test_cash_secured_put_reserved_cash_parses_symbol_when_strike_missing():
    position = _make_option_position(
        symbol="AAPL_041726P170",
        strike_price=None,
    )
    assert cash_secured_put_reserved_cash(position) == 17_000.0


def test_cash_secured_put_reserved_cash_ignores_long_puts_and_calls():
    long_put = _make_option_position(short_qty=0)
    long_put.longQuantity = 1
    long_put.instrument.putCall = "PUT"

    short_call = _make_option_position(short_qty=1)
    short_call.instrument.putCall = "CALL"

    assert cash_secured_put_reserved_cash(long_put) is None
    assert cash_secured_put_reserved_cash(short_call) is None


def test_total_csp_reserved_cash_sums_short_puts():
    positions = [
        _make_option_position(strike_price=170.0, short_qty=1),
        _make_option_position(strike_price=180.0, short_qty=1),
        _make_position(symbol="MSFT"),
    ]
    assert total_csp_reserved_cash(positions) == 35_000.0


def test_summarize_csp_cash_reserves_includes_available_cash():
    positions = [_make_option_position(strike_price=170.0, short_qty=1)]
    summary = summarize_csp_cash_reserves(positions, cash_balance=25_000.0)

    assert summary["totalReservedCash"] == 17_000.0
    assert summary["availableCashAfterReserves"] == 8_000.0
    assert len(summary["positions"]) == 1
    assert summary["positions"][0]["reservedCash"] == 17_000.0


def test_enrich_positions_table_shows_reserved_cash_for_short_puts():
    account = _make_account(liquidation_value=100_000.0)
    account.securitiesAccount.currentBalances.cashBalance = 25_000.0
    positions = [
        _make_position(symbol="MSFT"),
        _make_option_position(strike_price=170.0, short_qty=1),
    ]

    table = _enrich_positions_table(positions, account=account)

    assert "RESERVED_CASH" in table
    assert "17000.00" in table
    assert "TOTAL_CSP_RESERVED_CASH: 17000.0" in table


def test_account_summary_includes_csp_reserve_lines():
    account = _make_account(liquidation_value=100_000.0)
    account.securitiesAccount.currentBalances.cashBalance = 25_000.0
    positions = [_make_option_position(strike_price=170.0, short_qty=1)]

    summary = _build_account_summary(account, positions=positions)

    assert "Cash reserved for cash-secured puts: ~$17,000" in summary
    assert "Cash available after CSP reserves: ~$8,000" in summary
