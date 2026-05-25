from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction
from app.models.intelligence_models import MarketNewsItem, PortfolioDigest
from app.models.intelligence_models import ProactiveAlert
from app.models.portfolio_memory_models import MorningBrief, PortfolioChanges
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService


def test_render_email_includes_macro_and_changes():
    service = MorningBriefDeliveryService(
        app_user_adapter=MagicMock(),
        delivery_adapter=MagicMock(),
        email_adapter=MagicMock(),
        portfolio_analysis_service=MagicMock(),
        portfolio_service=MagicMock(),
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        portfolio_memory_service=MagicMock(),
    )

    brief = MorningBrief(
        generated_at=datetime.now(timezone.utc),
        macro_regime="VIX at 18.0",
        digest=PortfolioDigest(
            sector_weights=[],
            macro_news=[
                MarketNewsItem(headline="Fed holds rates steady", source="Reuters"),
            ],
            top_news=[],
            earnings_this_week=["AAPL"],
        ),
        changes=PortfolioChanges(
            summary="portfolio value +1.20%; largest weight shift: AAPL 15.0% → 22.0%",
        ),
        top_alerts=[
            ProactiveAlert(
                action=AnalysisAction.ASSIGNMENT_RISK,
                label="assignment risk",
                reason="Short put ITM",
                priority=1,
                symbol="AAPL",
            )
        ],
    )

    subject, text_body, html_body = service._render_email(
        recipient_name="Alex",
        brief=brief,
    )

    assert subject == "Your PowerPocket morning brief"
    assert "Alex" in text_body
    assert "VIX at 18.0" in text_body
    assert "Fed holds rates steady" in text_body
    assert "Reuters" in text_body
    assert "portfolio value +1.20%" in text_body
    assert "AAPL" in text_body
    assert "assignment risk" in text_body
    assert "VIX at 18.0" in html_body
