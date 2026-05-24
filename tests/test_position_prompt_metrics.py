from app.core.prompts import (
    _enrich_positions_table,
    _portfolio_liquidation_value,
    _position_pnl_pct,
    _position_weight_pct,
)
from app.models.schwab_models import (
    AggregatedBalance,
    CurrentBalances,
    InitialBalances,
    Instrument,
    Position,
    ProjectedBalances,
    SchwabAccounts,
    SecuritiesAccount,
)


def _make_instrument(symbol: str = "AAPL", asset_type: str = "EQUITY") -> Instrument:
    return Instrument(
        assetType=asset_type,
        cusip="037833100",
        symbol=symbol,
    )


def _make_position(
    *,
    symbol: str = "AAPL",
    long_qty: float = 100,
    short_qty: float = 0,
    avg: float = 170.0,
    market_value: float = 20000.0,
    pnl: float = 3000.0,
) -> Position:
    return Position(
        shortQuantity=short_qty,
        averagePrice=avg,
        currentDayProfitLoss=50.0,
        currentDayProfitLossPercentage=0.25,
        longQuantity=long_qty,
        settledLongQuantity=long_qty,
        settledShortQuantity=short_qty,
        instrument=_make_instrument(symbol=symbol),
        marketValue=market_value,
        maintenanceRequirement=0.0,
        averageLongPrice=avg if long_qty > 0 else None,
        longOpenProfitLoss=pnl if long_qty > 0 else None,
        averageShortPrice=avg if short_qty > 0 else None,
        shortOpenProfitLoss=pnl if short_qty > 0 else None,
        currentDayCost=0.0,
    )


def _make_account(liquidation_value: float = 100_000.0) -> SchwabAccounts:
    zero_fields = {
        "accruedInterest": 0.0,
        "availableFundsNonMarginableTrade": 0.0,
        "bondValue": 0.0,
        "buyingPower": 0.0,
        "cashBalance": 0.0,
        "cashAvailableForTrading": 0.0,
        "cashReceipts": 0.0,
        "dayTradingBuyingPower": 0.0,
        "dayTradingBuyingPowerCall": 0.0,
        "dayTradingEquityCall": 0.0,
        "equity": liquidation_value,
        "equityPercentage": 100.0,
        "liquidationValue": liquidation_value,
        "longMarginValue": 0.0,
        "longOptionMarketValue": 0.0,
        "longStockValue": 0.0,
        "maintenanceCall": 0.0,
        "maintenanceRequirement": 0.0,
        "margin": 0.0,
        "marginEquity": 0.0,
        "moneyMarketFund": 0.0,
        "mutualFundValue": 0.0,
        "regTCall": 0.0,
        "shortMarginValue": 0.0,
        "shortOptionMarketValue": 0.0,
        "shortStockValue": 0.0,
        "totalCash": 0.0,
        "isInCall": False,
        "pendingDeposits": 0.0,
        "marginBalance": 0.0,
        "shortBalance": 0.0,
        "accountValue": liquidation_value,
    }
    initial = InitialBalances(**zero_fields)
    current = CurrentBalances(
        accruedInterest=0.0,
        cashBalance=0.0,
        cashReceipts=0.0,
        longOptionMarketValue=0.0,
        liquidationValue=liquidation_value,
        longMarketValue=0.0,
        moneyMarketFund=0.0,
        savings=0.0,
        shortMarketValue=0.0,
        pendingDeposits=0.0,
        mutualFundValue=0.0,
        bondValue=0.0,
        shortOptionMarketValue=0.0,
        availableFunds=0.0,
        availableFundsNonMarginableTrade=0.0,
        buyingPower=0.0,
        buyingPowerNonMarginableTrade=0.0,
        dayTradingBuyingPower=0.0,
        equity=liquidation_value,
        equityPercentage=100.0,
        longMarginValue=0.0,
        maintenanceCall=0.0,
        maintenanceRequirement=0.0,
        marginBalance=0.0,
        regTCall=0.0,
        shortBalance=0.0,
        shortMarginValue=0.0,
        sma=0.0,
    )
    projected = ProjectedBalances(
        availableFunds=0.0,
        availableFundsNonMarginableTrade=0.0,
        buyingPower=0.0,
        dayTradingBuyingPower=0.0,
        dayTradingBuyingPowerCall=0.0,
        maintenanceCall=0.0,
        regTCall=0.0,
        isInCall=False,
        stockBuyingPower=0.0,
    )
    securities = SecuritiesAccount(
        type="MARGIN",
        accountNumber="123456789",
        roundTrips=0,
        isDayTrader=False,
        isClosingOnlyRestricted=False,
        pfcbFlag=False,
        positions=[],
        initialBalances=initial,
        currentBalances=current,
        projectedBalances=projected,
    )
    return SchwabAccounts(
        securitiesAccount=securities,
        aggregatedBalance=AggregatedBalance(
            currentLiquidationValue=liquidation_value,
            liquidationValue=liquidation_value,
        ),
    )


def test_position_pnl_pct_from_cost_basis():
    position = _make_position(long_qty=100, avg=170.0, pnl=3000.0)
    assert _position_pnl_pct(position) == 3000 / 17000 * 100


def test_position_weight_pct_uses_portfolio_liquidation_value():
    position = _make_position(market_value=20_000.0)
    account = _make_account(liquidation_value=100_000.0)
    portfolio_value = _portfolio_liquidation_value(account=account, positions=[position])

    assert portfolio_value == 100_000.0
    assert _position_weight_pct(position, portfolio_value) == 20.0


def test_enrich_positions_table_includes_precomputed_columns():
    position = _make_position(market_value=20_000.0, pnl=3000.0)
    account = _make_account(liquidation_value=100_000.0)

    table = _enrich_positions_table([position], account=account)

    assert "PNL_% | WEIGHT_%" in table
    assert "RESERVED_CASH" in table
    assert "+17.6%" in table
    assert "+20.0%" in table
    assert "PORTFOLIO_LIQUIDATION_VALUE: 100000.0" in table
