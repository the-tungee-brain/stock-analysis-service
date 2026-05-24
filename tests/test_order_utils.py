from app.broker.order_utils import (
    is_equity_leg,
    is_option_leg,
    option_premium_per_contract,
    option_total_premium,
    order_total_cash,
)
from app.models.schwab_order_models import Instrument, OrderLeg


def test_option_premium_scales_fill_price_by_100_shares():
    assert option_premium_per_contract(12.20) == 1220.0
    assert option_total_premium(12.20, 2) == 2440.0


def test_is_option_leg_uses_asset_type():
    leg = OrderLeg(
        orderLegType="OPTION",
        instrument=Instrument(assetType="OPTION", symbol="NVDA  250620C00180000"),
    )
    assert is_option_leg(leg) is True
    assert is_equity_leg(leg) is False


def test_is_equity_leg_never_treated_as_option():
    leg = OrderLeg(
        orderLegType="EQUITY",
        instrument=Instrument(
            assetType="EQUITY",
            symbol="NVDA",
            type="EQUITY",
        ),
    )
    assert is_equity_leg(leg) is True
    assert is_option_leg(leg) is False


def test_order_total_cash_equity_is_fill_times_shares_not_times_100():
    leg = OrderLeg(
        orderLegType="EQUITY",
        instrument=Instrument(assetType="EQUITY", symbol="NVDA", type="EQUITY"),
    )

    total = order_total_cash(leg, fill_price_per_share=120.0, quantity=10)

    assert total == 1200.0
    assert total != 120.0 * 100
    assert total != 120.0 * 10 * 100


def test_order_total_cash_option_uses_premium_math():
    leg = OrderLeg(
        orderLegType="OPTION",
        instrument=Instrument(assetType="OPTION", symbol="NVDA  250620C00180000"),
    )

    assert order_total_cash(leg, fill_price_per_share=12.20, quantity=1) == 1220.0
