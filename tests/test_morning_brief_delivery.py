from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction
from app.models.intelligence_models import MarketNewsItem, PortfolioDigest
from app.models.intelligence_models import ProactiveAlert
from app.models.portfolio_memory_models import MorningBrief, PortfolioChanges
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService


def _service(**overrides) -> MorningBriefDeliveryService:
    defaults = {
        "app_user_adapter": MagicMock(),
        "delivery_adapter": MagicMock(),
        "email_adapter": MagicMock(),
        "portfolio_analysis_service": MagicMock(),
        "portfolio_service": MagicMock(),
        "transaction_service": MagicMock(),
        "schwab_auth_service": MagicMock(),
        "portfolio_memory_service": MagicMock(),
    }
    defaults.update(overrides)
    return MorningBriefDeliveryService(**defaults)


def _sample_brief() -> MorningBrief:
    return MorningBrief(
        generated_at=datetime.now(timezone.utc),
        macro_regime="VIX at 18.0",
        digest=PortfolioDigest(
            sector_weights=[],
            macro_news=[
                MarketNewsItem(
                    headline="Fed holds rates steady",
                    source="Reuters",
                    url="https://example.com/fed-rates",
                ),
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


def test_prewarm_all_warms_users_without_email():
    user = MagicMock()
    user.identity_sub = "user-1"
    user.email = "a@example.com"

    service = _service()
    service.app_user_adapter.list_users_with_schwab.return_value = [user]
    service.build_for_user = MagicMock(return_value=_sample_brief())

    result = service.prewarm_all()

    assert result.attempted == 1
    assert result.warmed == 1
    assert result.skipped == 0
    assert result.failed == 0
    service.build_for_user.assert_called_once_with(
        user_id="user-1",
        refresh=True,
        persist=False,
    )
    service.email_adapter.send_email.assert_not_called()


def test_render_email_includes_macro_and_changes():
    service = _service()

    subject, text_body, html_body = service._render_email(
        recipient_name="Alex",
        brief=_sample_brief(),
    )

    assert subject == "Your Tomcrest morning brief"
    assert "Alex" in text_body
    assert "VIX at 18.0" in text_body
    assert "Fed holds rates steady" in text_body
    assert "Reuters" in text_body
    assert "https://example.com/fed-rates" in text_body
    assert "portfolio value +1.20%" in text_body
    assert "AAPL" in text_body
    assert "assignment risk" in text_body
    assert "Open Tomcrest" in text_body
    assert "— Tomcrest" in text_body
    assert "VIX at 18.0" in html_body
    assert 'href="https://example.com/fed-rates"' in html_body
