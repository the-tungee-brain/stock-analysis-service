from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_paper_trade_analytics_service
from app.models.momentum_breakout_performance_models import (
    PaperTradeBucketDto,
    PaperTradePerformanceBucketsResponse,
    PaperTradePerformanceMetaDto,
    PaperTradePerformanceSummaryResponse,
    PaperTradePerformanceTradesResponse,
    PaperTradeRecordDto,
    PaperTradeSummaryDto,
)
from app.services.strategy.paper_trade_analytics_service import (
    PaperTradeAnalyticsService,
    PaperTradeBucketMetrics,
    PaperTradeSummaryMetrics,
)
from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord

router = APIRouter()


def _meta() -> PaperTradePerformanceMetaDto:
    return PaperTradePerformanceMetaDto()


def _summary_dto(metrics: PaperTradeSummaryMetrics) -> PaperTradeSummaryDto:
    return PaperTradeSummaryDto(
        totalAlerts=metrics.total_alerts,
        triggeredAlerts=metrics.triggered_alerts,
        expiredAlerts=metrics.expired_alerts,
        winRate=metrics.win_rate,
        averageWin=metrics.average_win,
        averageLoss=metrics.average_loss,
        expectancy=metrics.expectancy,
        profitFactor=metrics.profit_factor,
        averageHoldingDays=metrics.average_holding_days,
        maxDrawdown=metrics.max_drawdown,
        currentOpenTrades=metrics.current_open_trades,
    )


def _bucket_dto(bucket: PaperTradeBucketMetrics) -> PaperTradeBucketDto:
    return PaperTradeBucketDto(
        key=bucket.key,
        tradeCount=bucket.trade_count,
        winRate=bucket.win_rate,
        expectancy=bucket.expectancy,
        profitFactor=bucket.profit_factor,
        averageReturnPct=bucket.average_return_pct,
    )


def _trade_dto(record: PaperTradePerformanceRecord) -> PaperTradeRecordDto:
    return PaperTradeRecordDto(
        alertId=record.alert_id,
        symbol=record.symbol,
        setupName=record.setup_name,
        signalDate=record.signal_date,
        entryTriggeredAt=record.entry_triggered_at,
        entryPrice=record.entry_price,
        stopPrice=record.stop_price,
        targetPrice=record.target_price,
        exitAt=record.exit_at,
        exitPrice=record.exit_price,
        status=record.status,
        outcomeReturnPct=record.outcome_return_pct,
        holdingDays=record.holding_days,
        riskGateAction=record.risk_gate_action,
        marketRegime=record.market_regime,
        volumeRatio=record.volume_ratio,
        rsPercentile=record.rs_percentile,
        createdAt=record.created_at,
    )


@router.get(
    "/strategy/momentum-breakout/performance/summary",
    response_model=PaperTradePerformanceSummaryResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_paper_performance_summary(
    user_id: str = Depends(get_current_user_id),
    analytics: PaperTradeAnalyticsService = Depends(get_paper_trade_analytics_service),
) -> PaperTradePerformanceSummaryResponse:
    summary = await asyncio.to_thread(analytics.summary, user_id)
    by_risk = await asyncio.to_thread(analytics.by_risk_gate_action, user_id)
    return PaperTradePerformanceSummaryResponse(
        meta=_meta(),
        summary=_summary_dto(summary),
        byRiskGate=[_bucket_dto(bucket) for bucket in by_risk],
    )


@router.get(
    "/strategy/momentum-breakout/performance/trades",
    response_model=PaperTradePerformanceTradesResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_paper_performance_trades(
    limit: int = 100,
    user_id: str = Depends(get_current_user_id),
    analytics: PaperTradeAnalyticsService = Depends(get_paper_trade_analytics_service),
) -> PaperTradePerformanceTradesResponse:
    trades = await asyncio.to_thread(analytics.list_trades, user_id, limit=limit)
    return PaperTradePerformanceTradesResponse(
        meta=_meta(),
        trades=[_trade_dto(record) for record in trades],
    )


@router.get(
    "/strategy/momentum-breakout/performance/by-symbol",
    response_model=PaperTradePerformanceBucketsResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_paper_performance_by_symbol(
    user_id: str = Depends(get_current_user_id),
    analytics: PaperTradeAnalyticsService = Depends(get_paper_trade_analytics_service),
) -> PaperTradePerformanceBucketsResponse:
    buckets = await asyncio.to_thread(analytics.by_symbol, user_id)
    return PaperTradePerformanceBucketsResponse(
        meta=_meta(),
        buckets=[_bucket_dto(bucket) for bucket in buckets],
    )


@router.get(
    "/strategy/momentum-breakout/performance/by-regime",
    response_model=PaperTradePerformanceBucketsResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_paper_performance_by_regime(
    user_id: str = Depends(get_current_user_id),
    analytics: PaperTradeAnalyticsService = Depends(get_paper_trade_analytics_service),
) -> PaperTradePerformanceBucketsResponse:
    buckets = await asyncio.to_thread(analytics.by_regime, user_id)
    return PaperTradePerformanceBucketsResponse(
        meta=_meta(),
        buckets=[_bucket_dto(bucket) for bucket in buckets],
    )
