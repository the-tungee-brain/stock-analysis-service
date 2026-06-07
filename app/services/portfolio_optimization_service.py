from __future__ import annotations

from dataclasses import dataclass

from app.broker.option_utils import instrument_asset_type_by_symbol
from app.broker.sector_labels import ETF_SECTOR_LABEL, MISC_SECTOR_LABEL
from app.models.intelligence_models import SectorWeight
from app.models.portfolio_memory_models import PortfolioSnapshotRecord
from app.models.portfolio_optimization_models import (
    PortfolioOptimizationBreakdown,
    PortfolioOptimizationBreakdownItem,
    PortfolioOptimizationDriver,
    PortfolioOptimizationResponse,
    PortfolioOptimizationSuggestion,
    PortfolioStockWeight,
)
from app.models.schwab_models import Position, SchwabAccounts

_BROAD_ETF_SYMBOLS = frozenset(
    {
        "SPY",
        "VOO",
        "VTI",
        "IVV",
        "QQQ",
        "QQQM",
        "DIA",
        "IWM",
        "SCHD",
        "VYM",
        "BND",
        "AGG",
        "VXUS",
        "VEA",
        "VWO",
    }
)


@dataclass(frozen=True)
class _OptimizationHoldingRow:
    symbol: str
    market_value: float
    asset_type: str | None = None


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _score_from_penalty(max_score: float, penalty: float) -> float:
    return round(_clamp(max_score - penalty, 0.0, max_score), 2)


class PortfolioOptimizationService:
    def build_from_snapshot(
        self,
        *,
        snapshot: PortfolioSnapshotRecord | None,
    ) -> PortfolioOptimizationResponse:
        if snapshot is None:
            return self._empty_response(
                data_gaps=["No cached portfolio snapshot is available."]
            )
        if not snapshot.positions or not snapshot.liquidation_value:
            return self._empty_response(
                data_gaps=["Cached portfolio snapshot has no holdings."]
            )

        rows: dict[str, _OptimizationHoldingRow] = {}
        for position in snapshot.positions:
            symbol = position.symbol.upper()
            existing = rows.get(symbol)
            market_value = abs(position.market_value)
            rows[symbol] = _OptimizationHoldingRow(
                symbol=symbol,
                market_value=(existing.market_value if existing else 0.0)
                + market_value,
                asset_type=(existing.asset_type if existing else position.asset_type),
            )

        return self._build_from_rows(
            rows=list(rows.values()),
            liquidation_value=snapshot.liquidation_value,
            cash_after_csp=snapshot.cash_balance or 0.0,
        )

    def build_portfolio_optimization(
        self,
        *,
        positions: list[Position],
        account: SchwabAccounts,
    ) -> PortfolioOptimizationResponse:
        balances = account.securitiesAccount.currentBalances
        return self._build_from_rows(
            rows=self._position_rows(positions),
            liquidation_value=balances.liquidationValue,
            cash_after_csp=balances.cashBalance,
        )

    def _build_from_rows(
        self,
        *,
        rows: list[_OptimizationHoldingRow],
        liquidation_value: float,
        cash_after_csp: float,
    ) -> PortfolioOptimizationResponse:
        if not rows or liquidation_value <= 0:
            return self._empty_response(data_gaps=["No current holdings are available."])

        stock_weights = self._stock_weights_from_rows(
            rows=rows,
            liquidation_value=liquidation_value,
        )
        if not stock_weights:
            return self._empty_response(data_gaps=["No current holdings are available."])

        sector_weights = self._local_sector_weights(
            rows=rows,
            liquidation_value=liquidation_value,
        )

        top_sector = sector_weights[0] if sector_weights else None
        etf_weight = next(
            (
                sector.weight_pct
                for sector in sector_weights
                if sector.sector == ETF_SECTOR_LABEL
            ),
            0.0,
        )
        top1 = stock_weights[0].portfolio_weight_pct
        top3 = sum(item.portfolio_weight_pct for item in stock_weights[:3])
        effective_names = self._effective_names(stock_weights)

        stock_score = self._stock_concentration_score(
            top1=top1,
            top3=top3,
            effective_names=effective_names,
        )
        sector_score = self._sector_concentration_score(
            top_sector_pct=top_sector.weight_pct if top_sector else 0.0
        )
        etf_score = self._etf_score(etf_weight)
        cash_after_csp_pct = (
            (cash_after_csp / liquidation_value) * 100.0
            if liquidation_value > 0
            else 0.0
        )
        cash_score = self._cash_score(cash_after_csp, liquidation_value)
        count_score = self._position_count_score(len(stock_weights))

        breakdown = PortfolioOptimizationBreakdown(
            stock_concentration=PortfolioOptimizationBreakdownItem(
                score=stock_score,
                max_score=30,
                status=self._status(stock_score, 30),
                summary=f"Largest name is {top1:.1f}%; top three are {top3:.1f}%.",
            ),
            sector_concentration=PortfolioOptimizationBreakdownItem(
                score=sector_score,
                max_score=25,
                status=self._status(sector_score, 25),
                summary=(
                    f"Largest sector is {top_sector.sector} at {top_sector.weight_pct:.1f}%."
                    if top_sector
                    else "Sector allocation is not available."
                ),
            ),
            etf_diversification=PortfolioOptimizationBreakdownItem(
                score=etf_score,
                max_score=15,
                status=self._status(etf_score, 15),
                summary=f"Broad ETF exposure is {etf_weight:.1f}%.",
            ),
            cash_allocation=PortfolioOptimizationBreakdownItem(
                score=cash_score,
                max_score=10,
                status=self._status(cash_score, 10),
                summary=f"Cash after CSP reserves is {cash_after_csp_pct:.1f}% of portfolio.",
            ),
            position_count=PortfolioOptimizationBreakdownItem(
                score=count_score,
                max_score=10,
                status=self._status(count_score, 10),
                summary=f"{len(stock_weights)} names; effective diversification is {effective_names:.1f} names.",
            ),
            correlation=PortfolioOptimizationBreakdownItem(
                score=None,
                max_score=10,
                status="unavailable",
                summary="Correlation scoring is not available yet.",
            ),
        )

        score = round(
            stock_score + sector_score + etf_score + cash_score + count_score
        )
        return PortfolioOptimizationResponse(
            diversification_score=int(_clamp(score)),
            rating=self._rating(score),
            stock_weights=stock_weights,
            sector_weights=sector_weights,
            breakdown=breakdown,
            top_drivers=self._drivers(
                top1=top1,
                distinct_symbols=len(stock_weights),
                top_sector=top_sector,
                etf_weight=etf_weight,
            )[:5],
            ranked_suggestions=self._suggestions(
                stock_weights=stock_weights,
                distinct_symbols=len(stock_weights),
                effective_names=effective_names,
                top_sector=top_sector,
                etf_weight=etf_weight,
            )[:5],
        )

    @staticmethod
    def _empty_response(
        *,
        data_gaps: list[str] | None = None,
    ) -> PortfolioOptimizationResponse:
        unavailable = PortfolioOptimizationBreakdownItem(
            score=None,
            max_score=0,
            status="unavailable",
            summary="Portfolio positions are not available.",
        )
        return PortfolioOptimizationResponse(
            diversification_score=0,
            rating="Poor",
            stock_weights=[],
            sector_weights=[],
            breakdown=PortfolioOptimizationBreakdown(
                stock_concentration=unavailable.model_copy(update={"max_score": 30}),
                sector_concentration=unavailable.model_copy(update={"max_score": 25}),
                etf_diversification=unavailable.model_copy(update={"max_score": 15}),
                cash_allocation=unavailable.model_copy(update={"max_score": 10}),
                position_count=unavailable.model_copy(update={"max_score": 10}),
                correlation=unavailable.model_copy(update={"max_score": 10}),
            ),
            data_gaps=data_gaps or [],
        )

    @staticmethod
    def _position_rows(positions: list[Position]) -> list[_OptimizationHoldingRow]:
        asset_types = instrument_asset_type_by_symbol(positions)
        rows: dict[str, _OptimizationHoldingRow] = {}
        for position in positions:
            instrument = position.instrument
            symbol = (
                instrument.underlyingSymbol or instrument.symbol
                if instrument.assetType == "OPTION"
                else instrument.symbol
            )
            symbol = (symbol or "").upper()
            if not symbol:
                continue
            existing = rows.get(symbol)
            asset_type = asset_types.get(symbol, instrument.assetType)
            rows[symbol] = _OptimizationHoldingRow(
                symbol=symbol,
                market_value=(existing.market_value if existing else 0.0)
                + abs(position.marketValue),
                asset_type=(existing.asset_type if existing else asset_type),
            )
        return list(rows.values())

    @staticmethod
    def _stock_weights_from_rows(
        *,
        rows: list[_OptimizationHoldingRow],
        liquidation_value: float,
    ) -> list[PortfolioStockWeight]:
        total_holdings_market_value = sum(row.market_value for row in rows)
        if liquidation_value <= 0 or total_holdings_market_value <= 0:
            return []
        weights = [
            PortfolioStockWeight(
                symbol=row.symbol,
                portfolio_weight_pct=round(
                    (row.market_value / liquidation_value) * 100.0, 2
                ),
                invested_weight_pct=round(
                    (row.market_value / total_holdings_market_value) * 100.0, 2
                ),
                weight_pct=round((row.market_value / liquidation_value) * 100.0, 2),
                market_value=round(row.market_value, 2),
                level=PortfolioOptimizationService._stock_level(
                    (row.market_value / liquidation_value) * 100.0
                ),
            )
            for row in rows
            if row.market_value > 0
        ]
        weights.sort(key=lambda item: item.portfolio_weight_pct, reverse=True)
        return weights

    @staticmethod
    def _stock_level(weight_pct: float) -> str:
        if weight_pct >= 50:
            return "critical"
        if weight_pct >= 30:
            return "high"
        if weight_pct >= 20:
            return "elevated"
        return "normal"

    @staticmethod
    def _stock_concentration_score(
        *,
        top1: float,
        top3: float,
        effective_names: float,
    ) -> float:
        penalty = max(top1 - 20, 0) * 0.45
        penalty += max(top3 - 55, 0) * 0.25
        penalty += max(6 - effective_names, 0) * 2.5
        return _score_from_penalty(30, penalty)

    @staticmethod
    def _sector_concentration_score(*, top_sector_pct: float) -> float:
        penalty = max(top_sector_pct - 35, 0) * 0.45
        return _score_from_penalty(25, penalty)

    @staticmethod
    def _etf_score(etf_weight: float) -> float:
        if etf_weight >= 30:
            return 15.0
        if etf_weight <= 0:
            return 0.0
        return round(_clamp((etf_weight / 30) * 15, 0, 15), 2)

    @staticmethod
    def _cash_score(cash_after_csp: float, liquidation: float) -> float:
        if liquidation <= 0:
            return 0.0
        cash_pct = (cash_after_csp / liquidation) * 100.0
        if 5 <= cash_pct <= 20:
            return 10.0
        if cash_pct < 5:
            return round(_clamp(cash_pct / 5 * 10, 0, 10), 2)
        return _score_from_penalty(10, (cash_pct - 20) * 0.25)

    @staticmethod
    def _position_count_score(count: int) -> float:
        if 8 <= count <= 25:
            return 10.0
        if count < 8:
            return round(_clamp((count / 8) * 10, 0, 10), 2)
        return _score_from_penalty(10, (count - 25) * 0.2)

    @staticmethod
    def _status(score: float | None, max_score: float) -> str:
        if score is None or max_score <= 0:
            return "unavailable"
        ratio = score / max_score
        if ratio >= 0.85:
            return "strong"
        if ratio >= 0.65:
            return "good"
        if ratio >= 0.4:
            return "watch"
        return "poor"

    @staticmethod
    def _rating(score: float) -> str:
        if score >= 85:
            return "Excellent"
        if score >= 70:
            return "Good"
        if score >= 55:
            return "Fair"
        if score >= 40:
            return "Weak"
        return "Poor"

    @staticmethod
    def _local_sector_weights(
        *,
        rows: list[_OptimizationHoldingRow],
        liquidation_value: float,
    ) -> list[SectorWeight]:
        if liquidation_value <= 0:
            return []

        stock_by_symbol = {row.symbol: row for row in rows}
        by_sector: dict[str, tuple[float, list[str]]] = {}
        for symbol, row in stock_by_symbol.items():
            sector = (
                ETF_SECTOR_LABEL
                if row.asset_type == "ETF" or symbol in _BROAD_ETF_SYMBOLS
                else MISC_SECTOR_LABEL
            )
            total, symbols = by_sector.get(sector, (0.0, []))
            by_sector[sector] = (total + row.market_value, symbols + [symbol])

        weights = [
            SectorWeight(
                sector=sector,
                weight_pct=round((value / liquidation_value) * 100.0, 2),
                symbols=sorted(symbols),
            )
            for sector, (value, symbols) in by_sector.items()
        ]
        weights.sort(key=lambda item: item.weight_pct, reverse=True)
        return weights

    @staticmethod
    def _drivers(
        *,
        top1: float,
        distinct_symbols: int,
        top_sector: SectorWeight | None,
        etf_weight: float,
    ) -> list[PortfolioOptimizationDriver]:
        drivers: list[PortfolioOptimizationDriver] = []
        if top1 >= 20:
            drivers.append(
                PortfolioOptimizationDriver(
                    category="stockConcentration",
                    title="Single-name concentration",
                    detail=f"Largest holding is {top1:.1f}% of the portfolio.",
                    impact_score=min(100, top1 * 1.5),
                )
            )
        if top_sector and top_sector.weight_pct >= 35:
            drivers.append(
                PortfolioOptimizationDriver(
                    category="sectorConcentration",
                    title="Sector concentration",
                    detail=f"{top_sector.sector} is {top_sector.weight_pct:.1f}% of the portfolio.",
                    impact_score=min(100, top_sector.weight_pct * 1.2),
                )
            )
        if etf_weight < 20:
            drivers.append(
                PortfolioOptimizationDriver(
                    category="etfDiversification",
                    title="Low broad ETF exposure",
                    detail=f"ETF exposure is {etf_weight:.1f}%.",
                    impact_score=min(100, 60 - etf_weight),
                )
            )
        if distinct_symbols < 8:
            drivers.append(
                PortfolioOptimizationDriver(
                    category="positionCount",
                    title="Low position count",
                    detail=f"Only {distinct_symbols} names are represented.",
                    impact_score=max(25, (8 - distinct_symbols) * 12),
                )
            )
        drivers.sort(key=lambda item: item.impact_score, reverse=True)
        return drivers

    @staticmethod
    def _suggestions(
        *,
        stock_weights: list[PortfolioStockWeight],
        distinct_symbols: int,
        effective_names: float,
        top_sector: SectorWeight | None,
        etf_weight: float,
    ) -> list[PortfolioOptimizationSuggestion]:
        suggestions: list[PortfolioOptimizationSuggestion] = []
        rank = 1

        top_stock = stock_weights[0] if stock_weights else None
        if top_stock and top_stock.portfolio_weight_pct >= 20:
            target = (
                50.0
                if top_stock.portfolio_weight_pct >= 70
                else 30.0
                if top_stock.portfolio_weight_pct >= 50
                else 20.0
            )
            suggestions.append(
                PortfolioOptimizationSuggestion(
                    rank=rank,
                    category="stockConcentration",
                    title=f"Reduce {top_stock.symbol} below {target:.0f}%",
                    why=f"{top_stock.symbol} is {top_stock.portfolio_weight_pct:.1f}% of portfolio value.",
                    action="Trim or avoid adding until the position falls below the target weight.",
                    impact_score=min(100, top_stock.portfolio_weight_pct * 1.4),
                    estimated_score_improvement=round(
                        min(
                            30,
                            max(top_stock.portfolio_weight_pct - target, 0) * 0.5,
                        ),
                        0,
                    ),
                    symbols=[top_stock.symbol],
                )
            )
            rank += 1

        if top_sector and top_sector.weight_pct >= 35:
            suggestions.append(
                PortfolioOptimizationSuggestion(
                    rank=rank,
                    category="sectorConcentration",
                    title=f"Reduce {top_sector.sector} exposure",
                    why=f"{top_sector.sector} represents {top_sector.weight_pct:.1f}% of portfolio value.",
                    action="Direct new capital away from this sector until exposure normalizes.",
                    impact_score=min(100, top_sector.weight_pct * 1.2),
                    estimated_score_improvement=round(
                        min(25, max(top_sector.weight_pct - 35, 0) * 0.3),
                        0,
                    ),
                    symbols=top_sector.symbols[:5],
                )
            )
            rank += 1

        if etf_weight < 30:
            suggestions.append(
                PortfolioOptimizationSuggestion(
                    rank=rank,
                    category="etfDiversification",
                    title="Increase broad-market ETF exposure",
                    why=f"ETF exposure is {etf_weight:.1f}%, leaving diversification dependent on single names.",
                    action="Use future contributions or trim proceeds for broad ETF exposure.",
                    impact_score=min(100, 70 - etf_weight),
                    estimated_score_improvement=round(
                        min(15, (30 - etf_weight) * 0.3),
                        0,
                    ),
                    symbols=[],
                )
            )
            rank += 1

        if distinct_symbols < 8:
            suggestions.append(
                PortfolioOptimizationSuggestion(
                    rank=rank,
                    category="positionCount",
                    title="Broaden position count",
                    why=f"The portfolio has {distinct_symbols} names and {effective_names:.1f} effective names.",
                    action="Add diversified exposure before adding more to the largest holdings.",
                    impact_score=max(25, (8 - distinct_symbols) * 12),
                    estimated_score_improvement=round(min(10, 8 - distinct_symbols), 0),
                    symbols=[],
                )
            )

        suggestions.sort(key=lambda item: item.impact_score, reverse=True)
        return [
            item.model_copy(update={"rank": index + 1})
            for index, item in enumerate(suggestions)
        ]

    @staticmethod
    def _effective_names(stock_weights: list[PortfolioStockWeight]) -> float:
        hhi = sum((item.portfolio_weight_pct / 100.0) ** 2 for item in stock_weights)
        return round((1.0 / hhi) if hhi > 0 else 0.0, 2)
