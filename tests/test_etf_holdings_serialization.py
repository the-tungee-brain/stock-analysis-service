from app.models.company_research_models import EtfHoldingItem, EtfHoldingsContext


def test_etf_holdings_context_serializes_camel_case_for_ios():
    context = EtfHoldingsContext(
        ticker="SCHD",
        total_holdings=104,
        sector_breakdown={"Financial Services": 22.5},
        holdings=[
            EtfHoldingItem(
                ticker="JPM",
                name="JPMorgan Chase & Co.",
                weight_pct=4.2,
                market_cap="$580B",
            )
        ],
        dividend_yield="3.45%",
        expense_ratio="0.06%",
    )

    payload = context.model_dump(mode="json", by_alias=True)

    assert "totalHoldings" in payload
    assert "sectorBreakdown" in payload
    assert "dividendYield" in payload
    assert "expenseRatio" in payload
    assert "total_holdings" not in payload
    assert "sector_breakdown" not in payload

    holding = payload["holdings"][0]
    assert holding["weightPct"] == 4.2
    assert holding["marketCap"] == "$580B"
    assert "weight_pct" not in holding
    assert "market_cap" not in holding
