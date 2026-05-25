from unittest.mock import AsyncMock, MagicMock
import asyncio

from app.models.strategy_models import (
    DividendStrategyConfig,
    InvestmentStrategy,
    StrategyStockPick,
    StrategyStockPickLLM,
    StrategyStockSuggestionsLLMResponse,
    UserInvestmentProfile,
    WheelStrategyConfig,
)
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.strategy.strategy_stock_suggestion_service import (
    StrategyStockSuggestionService,
)


def _wheel_profile(*, symbols: list[str] | None = None) -> UserInvestmentProfile:
    return UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        risk_tolerance="moderate",
        options_experience="beginner",
        income_vs_growth="balanced",
        wheel=WheelStrategyConfig(wheel_symbols=symbols or []),
    )


def test_supports_stock_suggestions_for_all_strategy_types():
    assert StrategyStockSuggestionService.supports_stock_suggestions(
        InvestmentStrategy.WHEEL
    )
    assert StrategyStockSuggestionService.supports_stock_suggestions(
        InvestmentStrategy.ETF_CORE
    )


def test_normalize_picks_deduplicates_and_excludes():
    picks = [
        StrategyStockPick(symbol="aapl", rationale="one", fit_score=0.9),
        StrategyStockPick(symbol="AAPL", rationale="dup", fit_score=0.8),
        StrategyStockPick(symbol="msft", rationale="two", fit_score=0.85),
        StrategyStockPick(symbol="tsla", rationale="skip", fit_score=0.7),
    ]
    normalized = StrategyStockSuggestionService._normalize_picks(
        picks,
        limit=2,
        exclude_symbols=["TSLA"],
    )
    assert [pick.symbol for pick in normalized] == ["AAPL", "MSFT"]


def test_profile_fingerprint_changes_with_preferences():
    profile_a = _wheel_profile()
    profile_b = _wheel_profile()
    profile_b = profile_b.model_copy(update={"risk_tolerance": "aggressive"})
    fp_a = StrategyStockSuggestionService.profile_fingerprint(
        profile_a, InvestmentStrategy.WHEEL, limit=5
    )
    fp_b = StrategyStockSuggestionService.profile_fingerprint(
        profile_b, InvestmentStrategy.WHEEL, limit=5
    )
    assert fp_a != fp_b


def test_openai_schema_marks_all_properties_required():
    from app.core.llm_json import openai_response_schema

    schema = openai_response_schema(StrategyStockSuggestionsLLMResponse)
    assert set(schema["required"]) == set(schema["properties"].keys())

    pick_schema = schema["$defs"]["StrategyStockPickLLM"]
    assert set(pick_schema["required"]) == set(pick_schema["properties"].keys())


def test_build_strategy_stock_suggestions_prompt_includes_preferences():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.DIVIDEND,
        risk_tolerance="conservative",
        income_vs_growth="income",
        dividend=DividendStrategyConfig(
            dividend_symbols=[],
            target_yield_pct=4.0,
            max_payout_ratio=70.0,
        ),
    )
    prompts = PromptEnrichmentService().build_strategy_stock_suggestions_prompt(
        profile,
        strategy=InvestmentStrategy.DIVIDEND,
        limit=3,
    )
    user_prompt = prompts[1]
    assert "dividend" in user_prompt
    assert "conservative" in user_prompt
    assert "4.0%" in user_prompt


def test_suggest_stocks_returns_ranked_picks():
    prompt_service = PromptEnrichmentService()
    llm_service = MagicMock()
    llm_service.generate_from_prompts = AsyncMock(
        return_value=StrategyStockSuggestionsLLMResponse(
            picks=[
                StrategyStockPickLLM(
                    symbol="ko",
                    companyName="Coca-Cola",
                    rationale="Stable dividend payer.",
                    fitScore=0.91,
                    tags=["dividend"],
                ),
                StrategyStockPickLLM(
                    symbol="JNJ",
                    companyName="Johnson & Johnson",
                    rationale="Defensive income name.",
                    fitScore=0.88,
                    tags=["healthcare"],
                ),
            ],
            summary="Conservative income-focused picks.",
        )
    )
    service = StrategyStockSuggestionService(prompt_service, llm_service)
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.DIVIDEND,
        dividend=DividendStrategyConfig(dividend_symbols=[]),
    )

    suggestions = asyncio.run(
        service.suggest_stocks(
            profile=profile,
            strategy=InvestmentStrategy.DIVIDEND,
            limit=2,
        )
    )

    assert suggestions is not None
    assert suggestions.picks[0].symbol == "KO"
    assert suggestions.summary == "Conservative income-focused picks."
    llm_service.generate_from_prompts.assert_awaited_once()


def test_suggest_stocks_excludes_existing_symbols():
    prompt_service = PromptEnrichmentService()
    llm_service = MagicMock()
    llm_service.generate_from_prompts = AsyncMock(
        return_value=StrategyStockSuggestionsLLMResponse(
            picks=[
                StrategyStockPickLLM(
                    symbol="AAPL",
                    companyName="Apple Inc.",
                    rationale="Already held.",
                    fitScore=0.95,
                    tags=[],
                ),
                StrategyStockPickLLM(
                    symbol="MSFT",
                    companyName="Microsoft",
                    rationale="Similar quality mega-cap.",
                    fitScore=0.9,
                    tags=[],
                ),
            ],
            summary="Additional ideas beyond your current list.",
        )
    )
    service = StrategyStockSuggestionService(prompt_service, llm_service)
    profile = _wheel_profile(symbols=["AAPL"])

    suggestions = asyncio.run(
        service.suggest_stocks(
            profile=profile,
            strategy=InvestmentStrategy.WHEEL,
            limit=2,
        )
    )

    assert suggestions is not None
    assert [pick.symbol for pick in suggestions.picks] == ["MSFT"]
