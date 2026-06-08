from unittest.mock import MagicMock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.service_dependencies import get_research_events_service
from app.main import app
from app.models.company_research_models import EarningsContext
from app.models.intelligence_models import EventTimelineEntry
from app.services.company_research_service import CompanyResearchService
from app.services.research_events_service import ResearchEventsService


def _auth_user():
    class _FakeUser:
        identity_sub = "user-1"

    return _FakeUser()


def test_research_events_route_does_not_call_company_research_build_context(
    monkeypatch,
):
    def _fail_build_context(*_args, **_kwargs):
        raise AssertionError("research events must not call full build_context")

    monkeypatch.setattr(CompanyResearchService, "build_context", _fail_build_context)

    research_events_service = MagicMock()
    research_events_service.get_events.return_value = [
        EventTimelineEntry(
            date="2026-05-20",
            kind="earnings",
            title="Earnings report (beat)",
            detail="EPS surprise +4.0%",
        )
    ]

    app.dependency_overrides[get_current_user] = _auth_user
    app.dependency_overrides[get_research_events_service] = (
        lambda: research_events_service
    )

    client = TestClient(app)
    try:
        response = client.get("/api/v1/research/events?symbol=nvda")
        assert response.status_code == 200
        assert response.json() == {
            "symbol": "NVDA",
            "events": [
                {
                    "date": "2026-05-20",
                    "kind": "earnings",
                    "title": "Earnings report (beat)",
                    "detail": "EPS surprise +4.0%",
                    "url": None,
                }
            ],
        }
        research_events_service.get_events.assert_called_once_with(symbol="NVDA")
    finally:
        app.dependency_overrides.clear()


def test_research_events_service_soft_fails_to_empty_events():
    sec_research_service = MagicMock()
    sec_research_service.filings.side_effect = RuntimeError("sec unavailable")
    earnings_service = MagicMock()
    earnings_service.build_research_context.side_effect = RuntimeError(
        "earnings unavailable"
    )
    service = ResearchEventsService(
        sec_research_service=sec_research_service,
        earnings_service=earnings_service,
    )

    assert service.get_events("NVDA") == []


def test_research_events_service_treats_missing_sec_cik_as_expected(caplog):
    sec_research_service = MagicMock()
    sec_research_service.filings.side_effect = HTTPException(
        status_code=404,
        detail="No SEC CIK found for symbol 'SCHD'",
    )
    earnings_service = MagicMock()
    earnings_service.build_research_context.return_value = None
    service = ResearchEventsService(
        sec_research_service=sec_research_service,
        earnings_service=earnings_service,
    )

    with caplog.at_level("INFO", logger="app.services.research_events_service"):
        events = service.get_events("SCHD")

    assert events == []
    assert "No SEC CIK found for symbol 'SCHD'" in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


def test_research_events_service_builds_public_earnings_event():
    sec_research_service = MagicMock()
    sec_research_service.filings.return_value = MagicMock(filings=[])
    earnings_service = MagicMock()
    earnings_service.build_research_context.return_value = EarningsContext(
        last_report_date="2026-05-20",
        last_beat_label="beat",
        last_eps_surprise_pct="+4.0%",
    )
    service = ResearchEventsService(
        sec_research_service=sec_research_service,
        earnings_service=earnings_service,
    )

    events = service.get_events("nvda")

    assert events == [
        EventTimelineEntry(
            date="2026-05-20",
            kind="earnings",
            title="Earnings report (beat)",
            detail="EPS surprise +4.0%",
        )
    ]
    sec_research_service.filings.assert_called_once_with(symbol="NVDA", limit=8)
    earnings_service.build_research_context.assert_called_once_with(symbol="NVDA")
