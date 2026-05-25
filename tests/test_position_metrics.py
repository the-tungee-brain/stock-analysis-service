import pytest

from app.broker.position_metrics import (
    annotate_position_metrics,
    position_open_profit_loss_pct,
    summarize_portfolio_metrics,
)
from app.models.schwab_models import Instrument, Position


def _option_position(
    *,
    avg: float = 10.0,
    market_value: float = 970.0,
    open_pl: float = -30.0,
) -> Position:
    return Position(
        shortQuantity=0.0,
        averagePrice=avg,
        currentDayProfitLoss=0.0,
        currentDayProfitLossPercentage=0.0,
        longQuantity=1.0,
        settledLongQuantity=1.0,
        settledShortQuantity=0.0,
        instrument=Instrument(
            assetType="OPTION",
            cusip="",
            symbol="AAPL  260620C00200000",
            putCall="CALL",
            underlyingSymbol="AAPL",
            strikePrice=200.0,
            expirationDate="2026-06-20",
        ),
        marketValue=market_value,
        maintenanceRequirement=0.0,
        averageLongPrice=avg,
        longOpenProfitLoss=open_pl,
        currentDayCost=0.0,
    )


def test_option_pnl_pct_uses_contract_multiplier():
    position = _option_position()
    assert position_open_profit_loss_pct(position) == pytest.approx(-3.0, abs=0.01)


def test_annotate_position_metrics_populates_computed_fields():
    position = _option_position()
    annotated = annotate_position_metrics(position, portfolio_value=100_000.0)

    assert annotated.openProfitLoss == pytest.approx(-30.0, abs=0.01)
    assert annotated.costBasis == pytest.approx(1000.0, abs=0.01)
    assert annotated.openProfitLossPct == pytest.approx(-3.0, abs=0.01)
    assert annotated.portfolioWeightPct == pytest.approx(0.97, abs=0.01)


def test_summarize_portfolio_metrics_aggregates_annotated_positions():
    positions = [
        annotate_position_metrics(_option_position(), portfolio_value=100_000.0),
        annotate_position_metrics(
            _option_position(market_value=5000.0, open_pl=500.0, avg=45.0),
            portfolio_value=100_000.0,
        ),
    ]

    summary = summarize_portfolio_metrics(positions)

    assert summary["totalOpenProfitLoss"] == pytest.approx(470.0, abs=0.01)
    assert summary["totalCostBasis"] == pytest.approx(5500.0, abs=0.01)
    assert summary["openProfitLossPct"] == pytest.approx(8.55, abs=0.01)
