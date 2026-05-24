from datetime import date, datetime, timedelta, timezone

from app.broker.order_grouping import (
    detect_roll_groups,
    format_option_contract_label,
    leg_contract_label,
    spread_group_for_order,
)
from app.broker.option_utils import parse_put_call_from_option_symbol
from app.models.schwab_order_models import (
    ExecutionLeg,
    Instrument,
    OrderActivity,
    OrderLeg,
    SchwabOrder,
)


def _make_option_order(
    *,
    order_id: int,
    underlying: str = "NVDA",
    instruction: str = "SELL_TO_OPEN",
    occ_symbol: str = "NVDA  250620C00180000",
    description: str = "NVDA 06/20/2025 180.00 C",
    fill_time: datetime | None = None,
    price: float = 2.5,
) -> SchwabOrder:
    fill_time = fill_time or datetime.now(timezone.utc) - timedelta(days=1)
    return SchwabOrder.model_construct(
        orderId=order_id,
        orderType="LIMIT",
        quantity=1,
        filledQuantity=1,
        status="FILLED",
        enteredTime=fill_time,
        closeTime=fill_time,
        orderLegCollection=[
            OrderLeg.model_construct(
                orderLegType="OPTION",
                legId=1,
                instruction=instruction,
                quantity=1,
                positionEffect="OPENING" if "OPEN" in instruction else "CLOSING",
                instrument=Instrument.model_construct(
                    symbol=occ_symbol,
                    description=description,
                    assetType="OPTION",
                    putCall="CALL" if "C" in occ_symbol else "PUT",
                ),
            )
        ],
        orderActivityCollection=[
            OrderActivity.model_construct(
                executionLegs=[
                    ExecutionLeg.model_construct(
                        legId=1,
                        price=price,
                        quantity=1,
                        time=fill_time,
                    )
                ]
            )
        ],
    )


def _make_vertical_spread(*, order_id: int = 100) -> SchwabOrder:
    fill_time = datetime.now(timezone.utc) - timedelta(days=1)
    return SchwabOrder.model_construct(
        orderId=order_id,
        orderType="NET_CREDIT",
        complexOrderStrategyType="VERTICAL",
        quantity=1,
        filledQuantity=1,
        status="FILLED",
        enteredTime=fill_time,
        closeTime=fill_time,
        orderLegCollection=[
            OrderLeg.model_construct(
                orderLegType="OPTION",
                legId=1,
                instruction="SELL_TO_OPEN",
                quantity=1,
                instrument=Instrument.model_construct(
                    symbol="NVDA  250620C00180000",
                    assetType="OPTION",
                    putCall="CALL",
                ),
            ),
            OrderLeg.model_construct(
                orderLegType="OPTION",
                legId=2,
                instruction="BUY_TO_OPEN",
                quantity=1,
                instrument=Instrument.model_construct(
                    symbol="NVDA  250620C00190000",
                    assetType="OPTION",
                    putCall="CALL",
                ),
            ),
        ],
        orderActivityCollection=[
            OrderActivity.model_construct(
                executionLegs=[
                    ExecutionLeg.model_construct(
                        legId=1, price=2.5, quantity=1, time=fill_time
                    ),
                    ExecutionLeg.model_construct(
                        legId=2, price=1.0, quantity=1, time=fill_time
                    ),
                ]
            )
        ],
    )


def test_parse_put_call_from_occ_symbol():
    assert parse_put_call_from_option_symbol("NVDA  250620C00180000") == "CALL"
    assert parse_put_call_from_option_symbol("AAPL_041726P170") == "PUT"


def test_format_option_contract_label():
    label = format_option_contract_label(
        expiration=date(2025, 6, 20),
        strike=180.0,
        put_call="CALL",
    )
    assert label == "Jun 20 '25 $180 Call"


def test_leg_contract_label_from_order_leg():
    order = _make_option_order(order_id=1)
    leg = order.orderLegCollection[0]
    assert leg_contract_label(leg) == "Jun 20 '25 $180 Call"


def test_spread_group_for_multi_leg_order():
    order = _make_vertical_spread()
    group = spread_group_for_order(order)
    assert group is not None
    assert group.kind == "spread"
    assert group.label == "Vertical spread"


def test_detect_roll_groups_same_day_close_and_open():
    fill_time = datetime.now(timezone.utc) - timedelta(days=1)
    close_order = _make_option_order(
        order_id=10,
        instruction="BUY_TO_CLOSE",
        occ_symbol="NVDA  250620C00180000",
        fill_time=fill_time,
    )
    open_order = _make_option_order(
        order_id=11,
        instruction="BUY_TO_OPEN",
        occ_symbol="NVDA  250718C00175000",
        description="NVDA 07/18/2025 175.00 C",
        fill_time=fill_time + timedelta(minutes=5),
    )

    groups = detect_roll_groups([close_order, open_order])

    assert 10 in groups
    assert 11 in groups
    assert groups[10].kind == "roll"
    assert "Roll:" in groups[10].label
    assert "180 Call" in groups[10].label
    assert "175 Call" in groups[10].label
