from app.broker.strategy_detector import detect_option_strategy
from app.models.schwab_models import Instrument, Position
from tests.test_option_utils import _make_option_position
from tests.test_position_prompt_metrics import _make_position


def test_detect_covered_call_when_shares_cover_contracts():
    stock = _make_position(symbol="AAPL")
    stock.longQuantity = 200
    option = _make_option_position(
        symbol="AAPL_041726C190",
        put_call="CALL",
        short_qty=1,
    )
    positions = [stock, option]

    assert detect_option_strategy(option, positions) == "covered_call"


def test_detect_naked_call_when_shares_insufficient():
    stock = _make_position(symbol="AAPL")
    stock.longQuantity = 50
    option = _make_option_position(
        symbol="AAPL_041726C190",
        put_call="CALL",
        short_qty=1,
    )

    assert detect_option_strategy(option, [stock, option]) == "naked_call"


def test_detect_cash_secured_put():
    option = _make_option_position(put_call="PUT", short_qty=2)
    assert detect_option_strategy(option, [option]) == "cash_secured_put"


def test_detect_long_call_and_long_put():
    long_call = _make_option_position(put_call="CALL", short_qty=0)
    long_call.longQuantity = 1

    long_put = _make_option_position(put_call="PUT", short_qty=0)
    long_put.longQuantity = 1

    assert detect_option_strategy(long_call, [long_call]) == "long_call"
    assert detect_option_strategy(long_put, [long_put]) == "long_put"


def test_detect_option_strategy_returns_none_for_equity():
    stock = _make_position(symbol="MSFT")
    assert detect_option_strategy(stock, [stock]) is None


def test_detect_unknown_when_put_call_missing():
    option = _make_option_position(put_call="PUT", short_qty=1)
    option.instrument.putCall = None
    assert detect_option_strategy(option, [option]) == "unknown"
