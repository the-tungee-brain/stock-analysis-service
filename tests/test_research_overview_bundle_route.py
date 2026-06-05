import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
from fastapi import Request

from app.api.get_research_overview_bundle_route import (
    get_research_overview_bundle,
    get_research_overview_enrichment,
    get_research_overview_fast,
)
from app.http.etag import json_weak_etag
from app.http.json_sanitizer import sanitize_json_value
from app.models.company_research_models import PerformanceSnapshot, ResearchSnapshot
from app.models.intelligence_models import SymbolIntelligence
from app.services.research_overview_service import ResearchOverviewBundle


def _sample_bundle() -> ResearchOverviewBundle:
    return ResearchOverviewBundle(
        symbol="AAPL",
        as_of=datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc),
        snapshot=ResearchSnapshot(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            country="US",
            price=200.0,
            changePct=0.5,
            marketCap="3.0T",
            weburl="https://apple.com",
            logo="https://example.com/logo.png",
        ),
        performance=PerformanceSnapshot(
            oneMonth="+2%",
            threeMonth="+5%",
            oneYear="+10%",
            trendLabel="Up",
            volatilityNote="Moderate",
        ),
        intelligence=SymbolIntelligence(symbol="AAPL", signals=[], eventTimeline=[]),
    )


def _mock_service(bundle: ResearchOverviewBundle) -> MagicMock:
    service = MagicMock()
    service.build_bundle_async = AsyncMock(return_value=bundle)
    service.build_fast_bundle_async = AsyncMock(return_value=bundle)
    service.build_enrichment_bundle_async = AsyncMock(return_value=bundle)
    return service


def test_overview_bundle_returns_etag_header():
    bundle = _sample_bundle()
    service = _mock_service(bundle)

    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )

    response = asyncio.run(
        get_research_overview_bundle(
            request=request,
            symbol="AAPL",
            holdings_limit=8,
            include_summary=False,
            user_id="user-1",
            overview_service=service,
        )
    )

    payload = bundle.model_dump(mode="json", by_alias=True)
    expected_etag = json_weak_etag(payload)
    assert response.headers["etag"] == f'"{expected_etag}"'
    assert response.status_code == 200


def test_overview_bundle_returns_304_when_etag_matches():
    bundle = _sample_bundle()
    service = _mock_service(bundle)

    payload = bundle.model_dump(mode="json", by_alias=True)
    etag = json_weak_etag(payload)

    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"if-none-match", f'"{etag}"'.encode())],
        }
    )

    response = asyncio.run(
        get_research_overview_bundle(
            request=request,
            symbol="AAPL",
            holdings_limit=8,
            include_summary=False,
            user_id="user-1",
            overview_service=service,
        )
    )

    assert response.status_code == 304
    service.build_bundle_async.assert_awaited_once()


def test_overview_fast_returns_valid_bundle_shape():
    bundle = _sample_bundle()
    bundle.intelligence.partial = True
    service = _mock_service(bundle)

    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )

    response = asyncio.run(
        get_research_overview_fast(
            request=request,
            symbol="AAPL",
            holdings_limit=8,
            user_id="user-1",
            overview_service=service,
        )
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["symbol"] == "AAPL"
    assert payload["snapshot"]["name"] == "Apple Inc."
    assert payload["intelligence"]["partial"] is True
    service.build_fast_bundle_async.assert_awaited_once_with(
        symbol="AAPL",
        holdings_limit=8,
    )
    service.build_bundle_async.assert_not_called()


def test_overview_enrichment_parses_requested_sections():
    bundle = _sample_bundle()
    service = _mock_service(bundle)

    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )

    response = asyncio.run(
        get_research_overview_enrichment(
            request=request,
            symbol="AAPL",
            holdings_limit=8,
            sections="street,intelligence",
            include_summary=False,
            user_id="user-1",
            overview_service=service,
        )
    )

    assert response.status_code == 200
    service.build_enrichment_bundle_async.assert_awaited_once_with(
        user_id="user-1",
        symbol="AAPL",
        holdings_limit=8,
        sections={"street", "intelligence"},
        include_summary=False,
    )
    service.build_bundle_async.assert_not_called()


def test_sanitize_json_value_converts_nested_non_finite_values_to_none():
    payload = {
        "price": float("nan"),
        "nested": {
            "upside": float("inf"),
            "downside": float("-inf"),
            "numpy": np.float64("nan"),
            "pandas": pd.NA,
        },
        "items": [1.0, np.float64("inf"), pd.NaT],
    }

    assert sanitize_json_value(payload) == {
        "price": None,
        "nested": {
            "upside": None,
            "downside": None,
            "numpy": None,
            "pandas": None,
        },
        "items": [1.0, None, None],
    }


def test_overview_bundle_sanitizes_non_finite_values_before_json_response():
    bundle = _sample_bundle()
    bundle.snapshot.price = float("nan")
    bundle.snapshot.changePct = float("inf")
    service = _mock_service(bundle)

    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )

    response = asyncio.run(
        get_research_overview_bundle(
            request=request,
            symbol="AAPL",
            holdings_limit=8,
            include_summary=False,
            user_id="user-1",
            overview_service=service,
        )
    )

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["snapshot"]["price"] is None
    assert payload["snapshot"]["changePct"] is None

    expected_payload = sanitize_json_value(
        bundle.model_dump(mode="json", by_alias=True)
    )
    assert response.headers["etag"] == f'"{json_weak_etag(expected_payload)}"'
