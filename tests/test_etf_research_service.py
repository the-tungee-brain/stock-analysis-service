from unittest.mock import MagicMock

import pytest

from app.adapters.securitiesdb.securitiesdb_adapter import SecuritiesDbAdapter
from app.services.etf_research_service import EtfResearchService
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.models.company_research_models import EtfHoldingsContext
from app.services.prompt_enrichment_service import PromptEnrichmentService


SPY_PAYLOAD = {
    "meta": {
        "confidence_score": 0.69,
        "domains": {
            "etf_holdings": {
                "last_updated": "2026-05-24T23:59:59.000Z",
            }
        },
    },
    "data": {
        "ticker": "SPY",
        "total_holdings": 504,
        "aum": 640_000_000_000,
        "sector_breakdown": {
            "Technology": 34.18,
            "Healthcare": 5.99,
        },
        "holdings": [
            {
                "ticker": "NVDA",
                "name": "NVIDIA CORP",
                "weight_pct": 8.3483,
                "sector": "Technology",
                "market_cap": 5_215_507_972_096,
                "piotroski_f": 4,
                "altman_z": 60.4029,
            },
            {
                "ticker": "AAPL",
                "name": "APPLE INC",
                "weight_pct": 7.0078,
                "sector": "Technology",
                "market_cap": 4_535_749_181_440,
                "piotroski_f": 8,
                "altman_z": 10.3117,
            },
            {
                "ticker": "C",
                "name": "CITIGROUP INC",
                "weight_pct": 0.3424,
                "sector": "Financial Services",
                "market_cap": 213_350_612_992,
                "piotroski_f": 2,
                "altman_z": 0.1988,
            },
        ],
    },
}


def test_etf_research_service_maps_holdings_payload():
    adapter = MagicMock()
    adapter.get_etf_holdings.return_value = SPY_PAYLOAD

    fundamentals = MagicMock()
    fundamentals.build_etf_metrics.return_value = {
        "dividend_yield": "1.25%",
        "expense_ratio": "0.09%",
    }

    context = EtfResearchService(
        securitiesdb_adapter=adapter,
        fundamentals_builder=fundamentals,
    ).build_holdings_context("SPY", holdings_limit=2)

    assert isinstance(context, EtfHoldingsContext)
    assert context.ticker == "SPY"
    assert context.total_holdings == 504
    assert context.aum == "$640.0B"
    assert context.sector_breakdown["Technology"] == pytest.approx(34.18)
    assert len(context.holdings) == 2
    assert context.holdings[0].ticker == "NVDA"
    assert context.holdings[0].weight_pct == pytest.approx(8.3483)
    assert context.holdings[0].market_cap == "$5.2T"
    assert context.holdings[0].piotroski_f == 4
    assert context.holdings[0].altman_z == pytest.approx(60.4029)
    assert context.holdings[0].quality_score is not None
    assert context.strongest_holdings[0].ticker == "AAPL"
    assert context.weakest_holdings[0].ticker == "C"
    assert context.dividend_yield == "1.25%"
    assert context.expense_ratio == "0.09%"
    assert context.data_as_of == "2026-05-24T23:59:59.000Z"
    assert context.confidence_score == pytest.approx(0.69)


def test_securitiesdb_adapter_returns_none_on_404():
    session = MagicMock()
    response = MagicMock()
    response.status_code = 404
    session.get.return_value = response

    result = SecuritiesDbAdapter(session=session, cache_ttl_seconds=60).get_etf_holdings(
        "UNKNOWN"
    )

    assert result is None


def test_research_context_block_includes_etf_holdings():
    etf = EtfHoldingsContext(
        ticker="SPY",
        total_holdings=504,
        aum="$640.0B",
        sector_breakdown={"Technology": 34.18},
        holdings=[
            {
                "ticker": "NVDA",
                "name": "NVIDIA CORP",
                "weight_pct": 8.35,
                "sector": "Technology",
            }
        ],
        expense_ratio="0.09%",
        dividend_yield="1.25%",
    )
    block = PromptEnrichmentService()._format_etf_holdings_section(etf)

    assert "ETF composition" in block
    assert "Sector breakdown" in block
    assert "Top holdings" in block
    assert "NVDA" in block
    assert "Technology: 34.18%" in block
