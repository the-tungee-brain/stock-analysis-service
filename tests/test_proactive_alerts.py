from app.core.prompts import AnalysisAction
from app.models.intelligence_models import IntelligenceSignal
from app.services.intelligence.signal_engine import build_proactive_alerts


def test_build_proactive_alerts_includes_assignment_risk():
    alerts = build_proactive_alerts(
        portfolio_signals=[],
        suggested_actions=[],
        earnings_this_week=[],
        assignment_risk_entries=[
            {
                "underlyingSymbol": "AAPL",
                "putCall": "PUT",
                "strike": 180.0,
                "daysToExpiration": 2,
                "moneyness": "ITM",
                "riskLevel": "critical",
            }
        ],
    )

    assert len(alerts) == 1
    assert alerts[0].action is AnalysisAction.ASSIGNMENT_RISK
    assert alerts[0].symbol == "AAPL"
    assert alerts[0].priority == 1


def test_build_proactive_alerts_includes_warning_concentration():
    alerts = build_proactive_alerts(
        portfolio_signals=[
            IntelligenceSignal(
                kind="concentration",
                severity="warning",
                message="AAPL is 22% of portfolio.",
                symbol="AAPL",
            )
        ],
        suggested_actions=[],
        earnings_this_week=[],
    )

    assert len(alerts) == 1
    assert alerts[0].action is AnalysisAction.CONCENTRATION_CHECK
    assert alerts[0].symbol == "AAPL"


def test_build_proactive_alerts_deduplicates_assignment_and_earnings():
    alerts = build_proactive_alerts(
        portfolio_signals=[
            IntelligenceSignal(
                kind="earnings",
                severity="warning",
                message="AAPL reports earnings in 3 days.",
                symbol="AAPL",
            )
        ],
        suggested_actions=[],
        earnings_this_week=["AAPL"],
    )

    assert len(alerts) == 1
    assert alerts[0].symbol == "AAPL"
