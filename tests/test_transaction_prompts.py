from datetime import datetime, timezone

from app.core.prompts import SymbolContext, build_symbol_prompt, AnalysisAction
from app.models.schwab_order_models import (
    ExecutionLeg,
    Instrument,
    OrderActivity,
    OrderLeg,
    SchwabOrder,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


def _make_filled_order(
    *,
    symbol: str = "NVDA",
    instruction: str = "BUY",
    quantity: float = 10,
    price: float = 120.0,
    fill_time: datetime | None = None,
) -> SchwabOrder:
    fill_time = fill_time or datetime(2026, 5, 20, 15, 30, tzinfo=timezone.utc)
    return SchwabOrder.model_construct(
        orderType="LIMIT",
        quantity=quantity,
        filledQuantity=quantity,
        status="FILLED",
        enteredTime=fill_time,
        closeTime=fill_time,
        taxLotMethod="FIFO",
        orderLegCollection=[
            OrderLeg.model_construct(
                orderLegType="EQUITY",
                legId=1,
                instruction=instruction,
                quantity=quantity,
                quantityType="SHARES",
                positionEffect="OPENING",
                instrument=Instrument.model_construct(
                    cusip="123",
                    symbol=symbol,
                    description=f"{symbol} INC",
                    instrumentId=1,
                    netChange=0.0,
                    type="EQUITY",
                ),
            )
        ],
        orderActivityCollection=[
            OrderActivity.model_construct(
                activityType="EXECUTION",
                executionType="FILL",
                quantity=quantity,
                orderRemainingQuantity=0.0,
                executionLegs=[
                    ExecutionLeg.model_construct(
                        legId=1,
                        price=price,
                        quantity=quantity,
                        mismarkedQuantity=0.0,
                        instrumentId=1,
                        time=fill_time,
                    )
                ],
            )
        ],
    )


def test_build_recent_transactions_markdown_formats_orders():
    orders = [
        _make_filled_order(instruction="SELL", quantity=5, price=130.0),
        _make_filled_order(
            instruction="BUY",
            quantity=10,
            price=120.0,
            fill_time=datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc),
        ),
    ]

    markdown = PromptEnrichmentService().build_recent_transactions_markdown(
        orders=orders,
        symbol="NVDA",
    )

    assert "Filled brokerage orders from the last 30 days" in markdown
    assert "SELL" in markdown
    assert "BUY" in markdown
    assert "$130.00" in markdown
    assert "$120.00" in markdown
    assert "FIFO" in markdown


def test_build_recent_transactions_markdown_empty():
    markdown = PromptEnrichmentService().build_recent_transactions_markdown(
        orders=[],
        symbol="NVDA",
    )

    assert "No filled orders for NVDA" in markdown


def test_build_symbol_prompt_includes_transactions_for_tax_angle():
    ctx = SymbolContext(
        symbol="NVDA",
        account=_make_account(),
        positions=[_make_position(symbol="NVDA")],
        recent_transactions="| Fill date | Side | Qty | Avg fill | Order type | Open/Close | Tax lot |",
        action=AnalysisAction.TAX_ANGLE,
    )

    prompt = build_symbol_prompt(ctx=ctx)

    assert "RECENT FILLED ORDERS" in prompt
    assert "Fill date" in prompt


def test_build_symbol_prompt_omits_transactions_for_free_form():
    ctx = SymbolContext(
        symbol="NVDA",
        account=_make_account(),
        positions=[_make_position(symbol="NVDA")],
        recent_transactions=None,
        action=AnalysisAction.FREE_FORM,
    )

    prompt = build_symbol_prompt(ctx=ctx)

    assert "RECENT FILLED ORDERS" not in prompt
