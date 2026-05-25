from unittest.mock import AsyncMock, MagicMock
import asyncio

from app.models.schwab_models import Instrument, Position
from app.models.strategy_models import (
    DividendStrategyConfig,
    EtfCoreStrategyConfig,
    InvestmentStrategy,
    JourneyStep,
    JourneyStepStatus,
    StrategyStockPick,
    StrategyStockPickLLM,
    StrategyStockSuggestionsLLMResponse,
    UserInvestmentProfile,
    UserStrategyJourney,
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


def _equity_position(symbol: str, *, shares: float = 100, market_value: float = 10000) -> Position:
    return Position(
        shortQuantity=0,
        averagePrice=100,
        currentDayProfitLoss=0,
        currentDayProfitLossPercentage=0,
        longQuantity=shares,
        settledLongQuantity=shares,
        settledShortQuantity=0,
        instrument=Instrument(
            assetType="EQUITY",
            cusip="123",
            symbol=symbol,
        ),
        marketValue=market_value,
        maintenanceRequirement=0,
        currentDayCost=0,
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
        journey_step_id="pick-names",
        journey_step_title="Pick dividend names",
        portfolio_context="Linked Schwab holdings (top positions by market value):\n- KO",
        macro_context="VIX at 18.0",
    )
    user_prompt = prompts[1]
    assert "dividend" in user_prompt
    assert "conservative" in user_prompt
    assert "4.0%" in user_prompt
    assert "Risk tolerance guidance" in user_prompt
    assert "Options experience guidance" not in user_prompt
    assert "Pick dividend names" in user_prompt
    assert "Portfolio context" in user_prompt
    assert "VIX at 18.0" in user_prompt


def test_build_strategy_stock_suggestions_prompt_includes_wheel_rubrics():
    profile = _wheel_profile()
    prompts = PromptEnrichmentService().build_strategy_stock_suggestions_prompt(
        profile,
        strategy=InvestmentStrategy.WHEEL,
        limit=3,
    )
    user_prompt = prompts[1]
    assert "Options experience guidance" in user_prompt
    assert "Target put delta" in user_prompt


def test_existing_symbols_are_strategy_scoped():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.DIVIDEND,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
        dividend=DividendStrategyConfig(dividend_symbols=["KO"]),
        etf_core=EtfCoreStrategyConfig(target_allocation={"VTI": 70.0}),
    )
    wheel_exclusions = StrategyStockSuggestionService._existing_symbols(
        profile,
        InvestmentStrategy.WHEEL,
    )
    dividend_exclusions = StrategyStockSuggestionService._existing_symbols(
        profile,
        InvestmentStrategy.DIVIDEND,
    )
    etf_exclusions = StrategyStockSuggestionService._existing_symbols(
        profile,
        InvestmentStrategy.ETF_CORE,
    )
    assert wheel_exclusions == ["AAPL"]
    assert dividend_exclusions == ["KO"]
    assert etf_exclusions == ["VTI"]


def test_existing_symbols_include_held_positions_for_covered_call():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.COVERED_CALL,
        wheel=WheelStrategyConfig(wheel_symbols=[]),
    )
    exclusions = StrategyStockSuggestionService._existing_symbols(
        profile,
        InvestmentStrategy.COVERED_CALL,
        held_symbols=["MSFT"],
    )
    assert exclusions == ["MSFT"]


def test_format_portfolio_context_includes_covered_call_ready_lots():
    context = StrategyStockSuggestionService.format_portfolio_context(
        [
            _equity_position("AAPL", shares=50, market_value=9000),
            _equity_position("MSFT", shares=150, market_value=45000),
        ],
        strategy=InvestmentStrategy.COVERED_CALL,
    )
    assert context is not None
    assert "MSFT" in context
    assert "100+ share lots available for covered calls: MSFT" in context


def test_resolve_journey_step_prefers_current_step_id():
    journey = UserStrategyJourney(
        user_id="user-1",
        strategy=InvestmentStrategy.WHEEL,
        current_step_id="research-underlying",
        steps=[
            JourneyStep(
                stepId="pick-underlying",
                title="Pick your underlying",
                description="Choose stocks",
                status=JourneyStepStatus.COMPLETED,
                order=1,
            ),
            JourneyStep(
                stepId="research-underlying",
                title="Research before you sell",
                description="Review risks",
                status=JourneyStepStatus.IN_PROGRESS,
                order=2,
            ),
        ],
        completion_pct=25.0,
    )
    step = StrategyStockSuggestionService.resolve_journey_step(journey)
    assert step is not None
    assert step.step_id == "research-underlying"


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
