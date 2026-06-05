from datetime import date, timedelta
from unittest.mock import MagicMock

from app.models.intelligence_models import SymbolIntelligence
from app.services.pattern_analysis_service import PatternAnalysisSnapshot
from app.services.portfolio_analysis_service import (
    INTELLIGENCE_OPTION_LOOKAHEAD_DAYS,
    INTELLIGENCE_OPTION_STRIKE_COUNT,
    PortfolioAnalysisService,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


def _prediction_payload(symbol: str = "AAPL") -> dict:
    return {
        "symbol": symbol,
        "date": "2026-06-04",
        "label_scheme": "binary_updown",
        "prediction": 1,
        "probabilities": {"0": 0.25, "1": 0.75},
        "up_prob": 0.75,
        "ranking_score": 0.75,
        "trade_signal": True,
        "in_training_universe": True,
        "indicators": {"rs_vs_spy_21d": 0.04, "new_high_52w": 1.0},
        "model_train_end_date": "2026-06-01",
        "model_key": "C",
        "model_label": "Relative strength + trend",
        "training_universe": "production",
        "n_features": 11,
        "feature_groups": ["relative_strength", "trend"],
    }


def _pattern_intelligence_payload(symbol: str = "AAPL") -> dict:
    return {
        "symbol": symbol,
        "as_of_date": "2026-06-04",
        "primary_pattern": None,
        "active_patterns": [],
        "trend_context": {
            "as_of_date": "2026-06-04",
            "close": 100.0,
            "sma50": 95.0,
            "sma200": 90.0,
            "above_sma50": True,
            "above_sma200": True,
            "trend_bias": "uptrend",
            "rs_vs_spy_21d": 0.04,
            "rs_vs_spy_63d": 0.08,
            "rs_vs_spy_126d": 0.12,
            "vol_ratio_20d": 1.5,
            "vol_zscore_20d": 0.6,
        },
        "scores": {
            "pattern_strength": 0.7,
            "trend_strength": 0.8,
            "relative_strength": 0.82,
            "volume_confirmation": 0.75,
            "model_alignment": 0.8,
            "confirmation_score": 0.78,
            "confidence": "high",
            "alignment_state": "confirmed",
        },
        "historical_stats": None,
        "setup_outcome": None,
        "core_model": _prediction_payload(symbol),
        "explanation": {
            "headline": "Constructive setup",
            "pattern_summary": "Pattern summary",
            "trend_context": "Trend context",
            "historical_context": "Historical context",
            "model_context": "Model context",
            "confidence_explanation": "Confidence explanation",
            "disclaimer": "Educational only",
        },
        "chart_intelligence": {
            "trendlines": [],
            "support_zones": [],
            "resistance_zones": [],
            "annotations": [],
            "highlighted_candles": [],
            "breakout_events": [],
            "fib_channel": None,
            "pattern_metadata": [],
            "summary": {
                "outlook": {
                    "label": "Constructive",
                    "tone": "positive",
                    "probability": 0.75,
                    "probability_display": "75%",
                    "expectation": "Upside bias",
                    "model_context": "Model agrees",
                    "is_benchmark": False,
                    "benchmark_notice": None,
                },
                "key_level": {
                    "label": "Support",
                    "price": 95.0,
                    "level_type": "support",
                    "display": "$95",
                    "implication": "Support held",
                    "available": True,
                },
                "why_this_outlook": [
                    {"text": "Trend acceleration is present.", "tone": "positive"}
                ],
                "thesis": "Constructive trend.",
                "disclaimer": "Educational only",
            },
        },
        "is_benchmark": False,
    }


def test_build_symbol_intelligence_returns_symbol_on_research_failure():
    company_research_service = MagicMock()
    company_research_service.build_context.side_effect = RuntimeError("fail")

    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )

    result = service.build_symbol_intelligence(
        user_id="user-1",
        symbol="AAPL",
    )

    assert result == SymbolIntelligence(symbol="AAPL", partial=True)


def test_build_symbol_intelligence_delegates_to_intelligence_service():
    ctx = MagicMock()
    company_research_service = MagicMock()
    company_research_service.build_context.return_value = ctx

    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.return_value = ctx
    expected = SymbolIntelligence(symbol="AAPL", signals=[])
    portfolio_intelligence_service.build_symbol_intelligence.return_value = expected

    market_service = MagicMock()
    market_service.get_option_chains.return_value = MagicMock()
    quote_snapshot = MagicMock()
    quote_snapshot.implied_vol = 0.285
    market_service.get_enriched_quote_snapshot.return_value = {"AAPL": quote_snapshot}

    transaction_service = MagicMock()
    transaction_service.get_filled_orders_by_symbol.return_value = []

    service = PortfolioAnalysisService(
        market_service=market_service,
        prompt_enrichment_service=MagicMock(),
        transaction_service=transaction_service,
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL")]

    result = service.build_symbol_intelligence(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=positions,
        access_token="token",
        include_options=True,
    )

    assert result == expected
    company_research_service.build_context.assert_called_once_with(symbol="AAPL")
    portfolio_intelligence_service.build_symbol_intelligence.assert_called_once()
    _, delegate_kwargs = (
        portfolio_intelligence_service.build_symbol_intelligence.call_args
    )
    assert delegate_kwargs["underlying_iv_percent"] == 0.285
    market_service.get_option_chains.assert_called_once()
    market_service.get_enriched_quote_snapshot.assert_called_once_with(
        access_token="token",
        symbols=["AAPL"],
    )
    _, kwargs = market_service.get_option_chains.call_args
    assert kwargs["strike_count"] == INTELLIGENCE_OPTION_STRIKE_COUNT
    today = date.today()
    assert kwargs["from_date"] == today.isoformat()
    assert kwargs["to_date"] == (
        today + timedelta(days=INTELLIGENCE_OPTION_LOOKAHEAD_DAYS)
    ).isoformat()


def test_build_symbol_intelligence_marks_partial_when_schwab_market_data_unavailable():
    ctx = MagicMock()
    company_research_service = MagicMock()
    company_research_service.build_context.return_value = ctx

    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.return_value = ctx
    portfolio_intelligence_service.build_symbol_intelligence.return_value = (
        SymbolIntelligence(symbol="C-PR", signals=[])
    )

    market_service = MagicMock()
    market_service.get_option_chains.return_value = None
    market_service.get_enriched_quote_snapshot.return_value = {}

    transaction_service = MagicMock()
    transaction_service.get_filled_orders_by_symbol.return_value = []

    service = PortfolioAnalysisService(
        market_service=market_service,
        prompt_enrichment_service=MagicMock(),
        transaction_service=transaction_service,
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )

    result = service.build_symbol_intelligence(
        user_id="user-1",
        symbol="C-PR",
        account=_make_account(),
        positions=[],
        access_token="token",
        include_options=True,
    )

    assert result.symbol == "C-PR"
    assert result.partial is True
    assert "schwab" in result.data_gaps


def test_build_symbol_intelligence_reuses_pattern_analysis_snapshot(monkeypatch):
    monkeypatch.setattr(
        "app.services.portfolio_analysis_service.is_paid_user",
        lambda _user_id: True,
    )

    def _unexpected_old_forecast(*_args, **_kwargs):
        raise AssertionError("old forecast builder should not be called")

    def _unexpected_old_intelligence(*_args, **_kwargs):
        raise AssertionError("old intelligence builder should not be called")

    monkeypatch.setattr(
        "app.services.pattern_forecast_service.build_pattern_trend_forecast",
        _unexpected_old_forecast,
    )
    monkeypatch.setattr(
        "app.services.pattern_intelligence_service.build_pattern_intelligence_payload",
        _unexpected_old_intelligence,
    )

    company_research_service = MagicMock()
    company_research_service.build_context.return_value = MagicMock()

    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.side_effect = lambda ctx: ctx
    portfolio_intelligence_service.build_symbol_intelligence.return_value = (
        SymbolIntelligence(symbol="AAPL", signals=[])
    )

    pattern_analysis_service = MagicMock()
    pattern_analysis_service.get_or_build.return_value = PatternAnalysisSnapshot(
        cache_key="pattern:AAPL",
        prediction_payload=_prediction_payload("AAPL"),
        pattern_intelligence=_pattern_intelligence_payload("AAPL"),
    )

    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
        pattern_analysis_service=pattern_analysis_service,
    )
    loaded_model = MagicMock()
    service.attach_pattern_model(loaded_model)

    result = service.build_symbol_intelligence(user_id="paid-user", symbol="AAPL")

    pattern_analysis_service.get_or_build.assert_called_once_with("AAPL", loaded_model)
    payload = result.model_dump(mode="json", by_alias=True)
    assert payload["patternForecast"]["upProb"] == 0.75
    assert payload["patternIntelligence"]["symbol"] == "AAPL"
    assert payload["patternIntelligence"]["trendContext"]["trendBias"] == "uptrend"
