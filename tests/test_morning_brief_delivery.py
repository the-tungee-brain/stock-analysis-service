from datetime import datetime, timezone
from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction
from app.models.intelligence_models import (
    HoldingCompanyNewsItem,
    MarketNewsItem,
    PortfolioDigest,
    SectorWeight,
)
from app.models.intelligence_models import ProactiveAlert
from app.models.portfolio_memory_models import (
    MorningBrief,
    MorningBriefMover,
    MorningBriefSnapshot,
    PortfolioChanges,
)
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
        snapshot=MorningBriefSnapshot(
            portfolio_value=68532,
            day_pnl=125,
            day_pnl_pct=0.18,
            cash_available=12360,
            diversification_score=24,
            diversification_rating="Poor",
            biggest_winner=MorningBriefMover(
                symbol="NVDA",
                day_pnl=320,
                day_pnl_pct=1.4,
            ),
            biggest_loser=MorningBriefMover(
                symbol="TSM",
                day_pnl=-180,
                day_pnl_pct=-2.1,
            ),
        ),
        macro_regime="VIX at 18.0; S&P 500 +0.42% today; Nasdaq -0.15% today; bonds bid",
        digest=PortfolioDigest(
            sector_weights=[
                SectorWeight(sector="Technology", weight_pct=68.0, symbols=["NVDA"])
            ],
            macro_news=[
                MarketNewsItem(
                    headline="Fed holds rates steady",
                    source="Reuters",
                    url="https://example.com/fed-rates",
                ),
            ],
            top_holdings_company_news=[
                HoldingCompanyNewsItem(
                    symbol="NVDA",
                    headline="NVDA supplier expansion may support AI demand",
                    source="Reuters",
                    summary="Reinforces long-term AI infrastructure spending.",
                    url="https://example.com/nvda",
                    weight_pct=51.2,
                )
            ],
            top_news=[],
            earnings_this_week=["AAPL"],
        ),
        changes=PortfolioChanges(
            liquidation_value_change=-41,
            liquidation_value_change_pct=-0.06,
            new_symbols=["MSFT"],
            removed_symbols=["NNE"],
            summary="portfolio value -0.06%",
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
    assert "Portfolio Snapshot" in text_body
    assert "Value: $68,532" in text_body
    assert "Day P/L: +$125 (+0.18%)" in text_body
    assert "Cash: $12,360" in text_body
    assert "Diversification Score: 24/100 (Poor)" in text_body
    assert "Top mover: NVDA +1.40%" in text_body
    assert "Weakest: TSM -2.10%" in text_body
    assert "Market Overview" in text_body
    assert "VIX at 18.0" in text_body
    assert "S&P 500 +0.42% today" in text_body
    assert "Nasdaq -0.15% today" in text_body
    assert "Portfolio Changes" in text_body
    assert "Added MSFT" in text_body
    assert "Removed NNE" in text_body
    assert "Portfolio News" in text_body
    assert "NVDA supplier expansion may support AI demand" in text_body
    assert "Why it matters: Reinforces long-term AI infrastructure spending." in text_body
    assert "Actionable Insight" in text_body
    assert "Reuters" in text_body
    assert "AAPL" in text_body
    assert "Risk Alerts" in text_body
    assert "assignment risk" in text_body
    assert "⚠️ Warning" in text_body
    assert "Open Tomcrest" in text_body
    assert "- Tomcrest" in text_body
    assert "Since yesterday" not in text_body
    assert "Market headlines" not in text_body
    assert "VIX at 18.0" in html_body
    assert 'href="https://example.com/nvda"' in html_body
