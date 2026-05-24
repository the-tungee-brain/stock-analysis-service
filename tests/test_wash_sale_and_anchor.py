from datetime import datetime, timedelta, timezone

from app.broker.order_grouping import detect_wash_sale_flags, last_fill_time_for_symbol
from app.core.prompts import AnalysisAction, _build_action_prompt
from app.models.company_research_models import NewsHeadline, ResearchContext
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.transaction_service import TransactionService
from tests.test_order_activity import _make_option_order
from tests.test_transaction_prompts import _make_filled_order


def test_detect_wash_sale_flags_equity_sell_then_buy():
    sell_time = datetime.now(timezone.utc) - timedelta(days=2)
    buy_time = datetime.now(timezone.utc) - timedelta(days=1)
    orders = [
        _make_filled_order(
            instruction="SELL",
            fill_time=sell_time,
            quantity=5,
            price=130.0,
        ),
        _make_filled_order(
            instruction="BUY",
            fill_time=buy_time,
            quantity=5,
            price=125.0,
        ),
    ]

    flags = detect_wash_sale_flags(orders, symbol="NVDA")

    assert len(flags) == 1
    assert flags[0].symbol == "NVDA"


def test_detect_wash_sale_flags_ignores_trades_outside_window():
    orders = [
        _make_filled_order(
            instruction="SELL",
            fill_time=datetime.now(timezone.utc) - timedelta(days=40),
        ),
        _make_filled_order(
            instruction="BUY",
            fill_time=datetime.now(timezone.utc) - timedelta(days=1),
        ),
    ]

    flags = detect_wash_sale_flags(orders, symbol="NVDA")

    assert flags == []


def test_suggest_analysis_actions_wash_sale_message():
    sell_time = datetime.now(timezone.utc) - timedelta(days=2)
    buy_time = datetime.now(timezone.utc) - timedelta(days=1)
    orders = [
        _make_filled_order(instruction="SELL", fill_time=sell_time),
        _make_filled_order(instruction="BUY", fill_time=buy_time),
    ]

    suggestions = TransactionService.suggest_analysis_actions(
        orders,
        symbol="NVDA",
    )

    tax = next(item for item in suggestions if item.action is AnalysisAction.TAX_ANGLE)
    assert "wash sale" in tax.reason.lower()


def test_last_fill_time_for_symbol():
    older = datetime.now(timezone.utc) - timedelta(days=5)
    newer = datetime.now(timezone.utc) - timedelta(days=1)
    orders = [
        _make_filled_order(fill_time=older),
        _make_filled_order(fill_time=newer),
    ]

    assert last_fill_time_for_symbol(orders, symbol="NVDA") == newer


def test_filter_news_since_last_fill():
    since = datetime(2026, 5, 10, tzinfo=timezone.utc)
    news = [
        NewsHeadline(
            headline="Old headline",
            summary="old",
            source="Reuters",
            datetime="2026-05-08T12:00:00+00:00",
        ),
        NewsHeadline(
            headline="New headline",
            summary="new",
            source="Bloomberg",
            datetime="2026-05-12T12:00:00+00:00",
        ),
    ]

    filtered = PromptEnrichmentService._filter_news_since(news, since)

    assert len(filtered) == 1
    assert filtered[0].headline == "New headline"


def test_research_context_block_news_heading_since_last_fill():
    since = datetime(2026, 5, 10, tzinfo=timezone.utc)
    ctx = ResearchContext(
        symbol="NVDA",
        news=[
            NewsHeadline(
                headline="Fresh headline",
                summary="summary",
                source="Reuters",
                datetime="2026-05-12T12:00:00+00:00",
            )
        ],
    )

    block = PromptEnrichmentService().format_research_context_block(
        ctx,
        action=AnalysisAction.WHAT_CHANGED,
        since=since,
    )

    assert "News since your last fill (May 10, 2026)" in block
    assert "Fresh headline" in block


def test_build_action_prompt_what_changed_includes_anchor():
    since = datetime(2026, 5, 10, tzinfo=timezone.utc)
    prompt = _build_action_prompt(
        AnalysisAction.WHAT_CHANGED,
        "NVDA",
        None,
        analysis_since=since,
    )

    assert "May 10, 2026" in prompt
    assert "since that fill" in prompt
