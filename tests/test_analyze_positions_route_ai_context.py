import json
import re

from fastapi.testclient import TestClient

from app.api.analyze_positions_by_symbol_route import AnalyzePositionsBySymbolRequest
from app.auth.dependencies import get_current_user, get_current_user_id
from app.core.prompts import AnalysisAction, PortfolioContext
from app.dependencies.service_dependencies import (
    get_ai_context_builder,
    get_chat_service,
    get_llm_service,
    get_portfolio_analysis_service,
    get_prompt_enrichment_service,
)
from app.main import app
from app.services.ai_context_builder import AIContextBuilder
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


class _FakePortfolioAnalysisService:
    async def build_analysis_context(self, **kwargs):
        return PortfolioContext(
            account=kwargs["account"],
            positions=kwargs["positions"],
            session_id=kwargs.get("session_id"),
            user_prompt=kwargs.get("user_prompt"),
            action=kwargs.get("action", AnalysisAction.FREE_FORM),
        )


class _FakeChatService:
    @staticmethod
    def user_message_for_storage(prompt, action):
        return (prompt or action.label).strip()

    @staticmethod
    def get_portfolio_analysis_session_id(**kwargs):
        return "session-id", True

    @staticmethod
    def get_chat_messages_by_session(session_id):
        return []

    @staticmethod
    def should_include_portfolio_context(**kwargs):
        return True

    @staticmethod
    def create_message(**kwargs):
        return None


class _CapturingLLMService:
    def __init__(self):
        self.calls = []

    async def analyze_option_position(self, **kwargs):
        self.calls.append(kwargs)
        yield (
            "I don't see an AAPL position in your portfolio. "
            "We can look at AAPL as stock analysis and a possible watchlist idea instead."
        )


def test_analyze_positions_route_includes_missing_position_context_and_safe_output():
    account = _make_account()
    positions = [_make_position(symbol="MSFT", market_value=12_000)]
    llm_service = _CapturingLLMService()

    app.dependency_overrides[get_current_user] = lambda: {"id": "user-1"}
    app.dependency_overrides[get_current_user_id] = lambda: "user-1"
    app.dependency_overrides[get_llm_service] = lambda: llm_service
    app.dependency_overrides[get_portfolio_analysis_service] = (
        lambda: _FakePortfolioAnalysisService()
    )
    app.dependency_overrides[get_prompt_enrichment_service] = (
        lambda: PromptEnrichmentService()
    )
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    app.dependency_overrides[get_ai_context_builder] = lambda: AIContextBuilder()

    request = AnalyzePositionsBySymbolRequest(
        account=account,
        positions=positions,
        prompt="What should I do with my AAPL position?",
        action=AnalysisAction.FREE_FORM,
        model="gpt-4.1-mini",
    )

    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/analyze-positions-by-symbol",
            json=request.model_dump(mode="json"),
            headers={"Authorization": "Bearer test-token"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.text
    assert "I don't see an AAPL position in your portfolio." in body
    assert not re.search(r"\b(hold|sell|trim|roll)\b", body, flags=re.IGNORECASE)

    assert len(llm_service.calls) == 1
    context_messages = llm_service.calls[0]["context_messages"]
    assert context_messages and context_messages[0]["role"] == "developer"
    developer_text = context_messages[0]["content"][0]["text"]
    assert "missing_position_rule" in developer_text
    assert "Never assume the user owns a symbol" in developer_text
    assert '"absent_position_symbols":["AAPL"]' in developer_text

    payload = developer_text.split("\n", 1)[1]
    context = json.loads(payload)
    ownership = context["strategy_policy"]["position_ownership"]
    assert ownership["requested_position_symbols"] == ["AAPL"]
    assert ownership["absent_position_symbols"] == ["AAPL"]
