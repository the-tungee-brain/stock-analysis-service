from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from app.broker.order_utils import order_relates_to_symbol
from app.core.prompts import AnalysisAction
from app.models.schwab_order_models import (
    ExecutionLeg,
    Instrument,
    OrderActivity,
    OrderLeg,
    SchwabOrder,
)
from app.services.transaction_service import TransactionService
from tests.test_transaction_prompts import _make_filled_order


def _make_option_order(*, underlying: str = "NVDA") -> SchwabOrder:
    fill_time = datetime.now(timezone.utc) - timedelta(days=2)
    occ_symbol = f"{underlying}  250620C00180000"
    return SchwabOrder.model_construct(
        orderType="LIMIT",
        quantity=1,
        filledQuantity=1,
        status="FILLED",
        enteredTime=fill_time,
        closeTime=fill_time,
        taxLotMethod="FIFO",
        orderLegCollection=[
            OrderLeg.model_construct(
                orderLegType="OPTION",
                legId=1,
                instruction="SELL_TO_OPEN",
                quantity=1,
                quantityType="CONTRACTS",
                positionEffect="OPENING",
                instrument=Instrument.model_construct(
                    cusip="123",
                    symbol=occ_symbol,
                    description=f"{underlying} 06/20/2025 180.00 C",
                    instrumentId=1,
                    netChange=0.0,
                    type="OPTION",
                    assetType="OPTION",
                ),
            )
        ],
        orderActivityCollection=[
            OrderActivity.model_construct(
                activityType="EXECUTION",
                executionType="FILL",
                quantity=1,
                orderRemainingQuantity=0.0,
                executionLegs=[
                    ExecutionLeg.model_construct(
                        legId=1,
                        price=2.5,
                        quantity=1,
                        mismarkedQuantity=0.0,
                        instrumentId=1,
                        time=fill_time,
                    )
                ],
            )
        ],
    )


def test_order_relates_to_symbol_matches_equity_and_option_underlying():
    equity_order = _make_filled_order(symbol="NVDA")
    option_order = _make_option_order(underlying="NVDA")

    assert order_relates_to_symbol(equity_order, "NVDA") is True
    assert order_relates_to_symbol(option_order, "NVDA") is True
    assert order_relates_to_symbol(equity_order, "AAPL") is False


def test_suggest_analysis_actions_after_recent_sell():
    orders = [
        _make_filled_order(
            instruction="SELL",
            fill_time=datetime.now(timezone.utc) - timedelta(days=1),
        )
    ]

    suggestions = TransactionService.suggest_analysis_actions(
        orders,
        symbol="NVDA",
    )

    actions = [item.action for item in suggestions]
    assert AnalysisAction.TAX_ANGLE in actions
    assert AnalysisAction.WHAT_CHANGED in actions


def test_suggest_analysis_actions_after_recent_buy():
    orders = [
        _make_filled_order(
            instruction="BUY",
            fill_time=datetime.now(timezone.utc) - timedelta(days=2),
        )
    ]

    suggestions = TransactionService.suggest_analysis_actions(
        orders,
        symbol="NVDA",
    )

    actions = [item.action for item in suggestions]
    assert AnalysisAction.WHAT_CHANGED in actions
    assert AnalysisAction.RISK_CHECK in actions
    assert AnalysisAction.TAX_ANGLE not in actions


def test_suggest_analysis_actions_ignores_stale_trades():
    orders = [
        _make_filled_order(
            instruction="SELL",
            fill_time=datetime.now(timezone.utc) - timedelta(days=20),
        )
    ]

    suggestions = TransactionService.suggest_analysis_actions(
        orders,
        symbol="NVDA",
    )

    assert suggestions == []


def test_to_recent_order_entry_equity_total_cash_without_multiplier():
    service = TransactionService(schwab_trader_builder=None)  # type: ignore[arg-type]
    entry = service.to_recent_order_entry(
        _make_filled_order(instruction="BUY", quantity=10, price=120.0)
    )

    assert entry.asset_type == "EQUITY"
    assert entry.premium_per_contract is None
    assert entry.total_premium is None
    assert entry.total_cash == 1200.0


def test_to_recent_order_entry_uses_underlying_for_options():
    service = TransactionService(schwab_trader_builder=None)  # type: ignore[arg-type]
    entry = service.to_recent_order_entry(_make_option_order(underlying="NVDA"))

    assert entry.symbol == "NVDA"
    assert entry.side == "SELL_TO_OPEN"
    assert entry.average_fill_price == 2.5
    assert entry.premium_per_contract == 250.0
    assert entry.total_premium == 250.0
    assert entry.asset_type == "OPTION"


def test_build_recent_transactions_markdown_equity_does_not_use_option_multiplier():
    from app.services.prompt_enrichment_service import PromptEnrichmentService

    orders = [
        _make_filled_order(instruction="BUY", quantity=10, price=120.0),
    ]
    markdown = PromptEnrichmentService().build_recent_transactions_markdown(
        orders=orders,
        symbol="NVDA",
    )

    assert "EQUITY rows" in markdown
    assert "10 sh" in markdown
    assert "$1,200.00" in markdown
    assert "Premium/contract (options only)" in markdown
    assert "| — |" in markdown or "| — | $1,200.00 |" in markdown.replace(" ", "")


def test_build_recent_transactions_markdown_includes_option_premium():
    from app.services.prompt_enrichment_service import PromptEnrichmentService

    markdown = PromptEnrichmentService().build_recent_transactions_markdown(
        orders=[_make_option_order(underlying="NVDA")],
        symbol="NVDA",
    )

    assert "OPTION rows only" in markdown
    assert "$250.00" in markdown
    assert "$2.50/sh" in markdown
    assert "$1,220 total cash" in markdown or "$1,220" in markdown


def test_build_recent_activity_summary():
    now = datetime.now(timezone.utc)
    orders = [
        _make_filled_order(
            symbol="NVDA",
            instruction="BUY",
            fill_time=now - timedelta(days=1),
        ),
        _make_filled_order(
            symbol="AAPL",
            instruction="SELL",
            fill_time=now - timedelta(days=3),
        ),
    ]

    builder = MagicMock()
    builder.get_orders = MagicMock(return_value=orders)
    service = TransactionService(schwab_trader_builder=builder)

    summary = service.build_recent_activity_summary(
        account_number="123",
        access_token="token",
        days_back=30,
    )

    assert summary.total_orders == 2
    assert summary.recent_order_count == 2
    assert len(summary.latest_orders) == 2
    assert summary.symbols_traded[0].symbol in {"NVDA", "AAPL"}
    assert AnalysisAction.WHAT_CHANGED in [item.action for item in summary.suggested_actions]


def test_fetch_filled_orders_uses_cache():
    now = datetime.now(timezone.utc)
    cached_orders = [
        _make_filled_order(fill_time=now - timedelta(days=1)),
    ]
    cache = MagicMock()
    cache.get = MagicMock(return_value=cached_orders)
    cache.put = MagicMock()

    builder = MagicMock()
    builder.get_orders = MagicMock()

    service = TransactionService(
        schwab_trader_builder=builder,
        recent_orders_cache=cache,
    )

    orders = service.get_filled_orders(
        account_number="123",
        access_token="token",
        user_id="user-1",
        days_back=30,
    )

    assert orders == cached_orders
    builder.get_orders.assert_not_called()
    cache.get.assert_called_once_with(
        user_id="user-1",
        account_number="123",
        days_back=30,
    )


def test_fetch_filled_orders_refresh_bypasses_cache():
    now = datetime.now(timezone.utc)
    fresh_orders = [
        _make_filled_order(fill_time=now - timedelta(days=1)),
    ]
    cache = MagicMock()
    cache.get = MagicMock(return_value=[_make_filled_order()])
    cache.put = MagicMock()
    cache.delete = MagicMock()

    builder = MagicMock()
    builder.get_orders = MagicMock(return_value=fresh_orders)

    service = TransactionService(
        schwab_trader_builder=builder,
        recent_orders_cache=cache,
    )

    orders = service.get_filled_orders(
        account_number="123",
        access_token="token",
        user_id="user-1",
        days_back=30,
        refresh=True,
    )

    assert orders == fresh_orders
    cache.delete.assert_called_once_with(
        user_id="user-1",
        account_number="123",
        days_back=30,
    )
    cache.get.assert_not_called()
    builder.get_orders.assert_called_once()


def test_invalidate_recent_orders_cache():
    cache = MagicMock()
    cache.invalidate_user = MagicMock(return_value=2)

    service = TransactionService(
        schwab_trader_builder=MagicMock(),
        recent_orders_cache=cache,
    )

    deleted = service.invalidate_recent_orders_cache(user_id="user-1")

    assert deleted == 2
    cache.invalidate_user.assert_called_once_with(user_id="user-1")
