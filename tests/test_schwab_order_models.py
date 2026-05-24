from datetime import datetime, timezone

from app.models.schwab_order_models import (
    ExecutionLeg,
    Instrument,
    OrderActivity,
    OrderLeg,
    SchwabOrder,
)


def test_schwab_order_validates_minimal_live_option_payload():
    payload = {
        "session": "NORMAL",
        "duration": "DAY",
        "orderType": "LIMIT",
        "complexOrderStrategyType": "NONE",
        "quantity": 1.0,
        "filledQuantity": 1.0,
        "remainingQuantity": 0.0,
        "orderStrategyType": "SINGLE",
        "orderId": 987654321,
        "cancelable": False,
        "editable": False,
        "status": "FILLED",
        "enteredTime": "2026-05-20T15:30:00+0000",
        "closeTime": "2026-05-20T15:30:01+0000",
        "orderLegCollection": [
            {
                "orderLegType": "OPTION",
                "legId": 1,
                "instrument": {
                    "assetType": "OPTION",
                    "symbol": "NVDA  250620C00180000",
                    "description": "NVDA 06/20/2025 180.00 C",
                    "instrumentId": 250138458,
                    "type": "VANILLA",
                    "putCall": "CALL",
                    "underlyingSymbol": "NVDA",
                    "optionDeliverableUnits": 100.0,
                },
                "instruction": "SELL_TO_OPEN",
                "positionEffect": "OPENING",
                "quantity": 1.0,
            }
        ],
        "orderActivityCollection": [
            {
                "activityType": "EXECUTION",
                "executionType": "FILL",
                "quantity": 1.0,
                "orderRemainingQuantity": 0.0,
                "executionLegs": [
                    {
                        "legId": 1,
                        "price": 2.5,
                        "quantity": 1.0,
                        "instrumentId": 250138458,
                        "time": "2026-05-20T15:30:01+0000",
                    }
                ],
            }
        ],
    }

    order = SchwabOrder.model_validate(payload)

    assert order.status == "FILLED"
    assert order.orderLegCollection is not None
    assert order.orderLegCollection[0].instruction == "SELL_TO_OPEN"
    assert order.orderLegCollection[0].instrument is not None
    assert order.orderLegCollection[0].instrument.assetType == "OPTION"
    assert order.orderLegCollection[0].instrument.underlyingSymbol == "NVDA"
    assert order.orderActivityCollection is not None
    assert order.orderActivityCollection[0].executionLegs is not None
    assert order.orderActivityCollection[0].executionLegs[0].price == 2.5


def test_schwab_order_roundtrip_json():
    order = SchwabOrder(
        orderType="LIMIT",
        quantity=10,
        filledQuantity=10,
        status="FILLED",
        enteredTime=datetime(2026, 5, 20, 15, 30, tzinfo=timezone.utc),
        closeTime=datetime(2026, 5, 20, 15, 30, tzinfo=timezone.utc),
    )

    restored = SchwabOrder.model_validate_json(order.model_dump_json())

    assert restored.orderType == "LIMIT"
    assert restored.status == "FILLED"
