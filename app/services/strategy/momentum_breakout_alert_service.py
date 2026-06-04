"""Educational Momentum Breakout trade-plan alerts with risk gating."""

from __future__ import annotations

from data.benchmarks import BENCHMARK_SYMBOL
from data.loader import load_symbol
from trade_planner.alerts.engine import AlertEngine
from trade_planner.alerts.risk_gate import AlertRiskGate
from trade_planner.alerts.risk_models import (
    AlertRiskContext,
    AlertRiskSettings,
    ClosedTradeSnapshot,
    OpenTradeSnapshot,
)
from trade_planner.config import TradePlannerConfig
from trade_planner.research.data import align_benchmark_to_stock, ohlcv_bars_from_dataframe
from trade_planner.research.features import capture_momentum_feature_snapshot
from trade_planner.research.models import MarketRegime
from trade_planner.research.regime import classify_market_regime
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import DuplicateActiveMomentumAlertError
from trade_planner.alerts.risk_models import AlertGateAction
from app.services.strategy.momentum_breakout_notification_emitter import (
    MomentumBreakoutNotificationEmitter,
)
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.service import TradePlannerService
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import StockData
from app.models.momentum_breakout_alert_models import (
    AlertRiskGateResultDto,
    AlertRiskSettingsDto,
    ClosedTradeSnapshotDto,
    HistoricalStatsDto,
    MomentumBreakoutAlertRequest,
    MomentumBreakoutAlertResponse,
    OpenTradeSnapshotDto,
    TradePlanLevelsDto,
)

DISCLAIMER = (
    "Educational trade plan information only. Not investment advice. "
    "Tomcrest does not place orders or manage your portfolio."
)


class MomentumBreakoutAlertService:
    def __init__(
        self,
        *,
        planner_service: TradePlannerService | None = None,
        risk_gate: AlertRiskGate | None = None,
        alert_engine: AlertEngine | None = None,
        lifecycle_service: AlertLifecycleService | None = None,
        notification_emitter: MomentumBreakoutNotificationEmitter | None = None,
    ) -> None:
        self._planner = planner_service or TradePlannerService(
            statistics_store=None,
        )
        self._risk_gate = risk_gate or AlertRiskGate()
        self._alert_engine = alert_engine or AlertEngine()
        self._lifecycle = lifecycle_service or AlertLifecycleService()
        self._emitter = notification_emitter
        self._setup = MomentumBreakoutSetup(TradePlannerConfig().momentum)

    @property
    def lifecycle_service(self) -> AlertLifecycleService:
        return self._lifecycle

    def evaluate(
        self,
        request: MomentumBreakoutAlertRequest,
        *,
        user_id: str | None = None,
    ) -> MomentumBreakoutAlertResponse:
        symbol = request.symbol.strip().upper()
        stock_df = load_symbol(symbol)
        bench_df = load_symbol(BENCHMARK_SYMBOL)
        stock_bars = ohlcv_bars_from_dataframe(stock_df)
        bench_bars = align_benchmark_to_stock(
            stock_bars, ohlcv_bars_from_dataframe(bench_df)
        )
        data = StockData.from_bars(symbol, stock_bars, benchmark_bars=bench_bars)

        plan = self._setup.build_plan(data)
        if plan is None:
            return MomentumBreakoutAlertResponse(
                disclaimer=DISCLAIMER,
                planAvailable=False,
                plan=None,
                historicalStats=None,
                riskGate=AlertRiskGateResultDto(
                    allowed=False,
                    action="BLOCK",
                    reasons=[
                        "No valid Momentum Breakout setup on latest bar.",
                        "Educational only — not investment advice.",
                    ],
                    recommendedPositionRiskPct=0.0,
                    maxNotionalUsd=None,
                    alertPriority="LOW",
                    educationalOnly=True,
                ),
                priceAlerts=[],
            )

        engine = BacktestEngine(self._planner.planner_config.backtest, store=None)
        result = engine.run(
            self._setup,
            stock_bars,
            symbol=symbol,
            benchmark_bars=bench_bars,
        )
        stats = result.statistics
        plan = plan.with_historical_statistics(stats)

        signal_index = len(stock_bars) - 1
        snapshot = capture_momentum_feature_snapshot(
            stock_bars,
            bench_bars,
            signal_index=signal_index,
            config=self._setup._config,  # noqa: SLF001
        )
        regime = snapshot.market_regime
        volume_ratio = snapshot.volume_ratio

        settings = self._map_settings(request.risk_settings)
        context = AlertRiskContext(
            candidate_plan=plan,
            open_trades=self._map_open(request.open_trades),
            recent_closed=self._map_closed(request.recent_closed),
            current_symbol=symbol,
            sector_or_group=request.sector_or_group,
            market_regime=self._parse_regime(request.market_regime) or regime,
            volume_ratio=request.volume_ratio if request.volume_ratio is not None else volume_ratio,
            account_equity_usd=request.account_equity_usd,
            settings=settings,
        )
        decision = self._risk_gate.evaluate(context)

        if user_id and self._emitter is not None and not decision.allowed:
            self._emitter.on_risk_gate_blocked(
                user_id,
                symbol=symbol,
                setup_name=plan.setup_name,
                reasons=tuple(decision.reasons),
                entry_price=plan.entry_price,
                stop_price=plan.stop_price,
                target_price=plan.target_price,
                risk_gate_action=decision.action.value,
            )

        price_alerts = self._alert_engine.evaluate(
            plan=plan,
            stock_data=data,
            prior_price=request.prior_price,
        )

        alert_id: str | None = None
        lifecycle_status: str | None = None
        if user_id and request.persist_alert and decision.allowed:
            try:
                record = AlertLifecycleService.build_record(
                    user_id=user_id,
                    symbol=symbol,
                    signal_date=stock_bars[-1].trading_date,
                    entry_price=plan.entry_price,
                    stop_price=plan.stop_price,
                    target_price=plan.target_price,
                    entry_is_stop=plan.entry_is_stop,
                    risk_gate_action=decision.action.value,
                    risk_gate_reasons=decision.reasons,
                    historical_win_rate=stats.win_rate if stats.total_trades else None,
                    historical_profit_factor=stats.profit_factor
                    if stats.total_trades
                    else None,
                    historical_total_trades=stats.total_trades or None,
                    market_regime=regime.value,
                    volume_ratio=volume_ratio,
                    rs_percentile=snapshot.rs_percentile,
                )
                created = self._lifecycle.create_alert(record)
                alert_id = created.alert_id
                lifecycle_status = created.status.value
                if (
                    self._emitter is not None
                    and decision.action
                    in {AlertGateAction.WARN, AlertGateAction.SIZE_DOWN}
                ):
                    self._emitter.on_risk_gate_warning(
                        user_id,
                        symbol=symbol,
                        setup_name=plan.setup_name,
                        reasons=tuple(decision.reasons),
                        entry_price=plan.entry_price,
                        stop_price=plan.stop_price,
                        target_price=plan.target_price,
                        risk_gate_action=decision.action.value,
                        alert_id=alert_id,
                    )
            except DuplicateActiveMomentumAlertError as exc:
                price_alerts = [
                    *price_alerts,
                    f"Lifecycle: {exc}",
                ]

        return MomentumBreakoutAlertResponse(
            disclaimer=DISCLAIMER,
            planAvailable=True,
            plan=TradePlanLevelsDto(
                symbol=plan.symbol,
                setupName=plan.setup_name,
                direction=plan.direction,
                entryPrice=plan.entry_price,
                stopPrice=plan.stop_price,
                targetPrice=plan.target_price,
                riskReward=plan.risk_reward,
                confidenceScore=plan.confidence_score,
                entryIsStop=plan.entry_is_stop,
            ),
            historicalStats=HistoricalStatsDto(
                totalTrades=stats.total_trades,
                winRatePct=stats.historical_win_rate_pct,
                profitFactor=stats.profit_factor,
                averageHoldingDays=stats.average_holding_days,
            )
            if stats.total_trades > 0
            else None,
            riskGate=AlertRiskGateResultDto(
                allowed=decision.allowed,
                action=decision.action.value,
                reasons=list(decision.reasons),
                recommendedPositionRiskPct=decision.recommended_position_risk_pct,
                maxNotionalUsd=decision.max_shares_or_dollars,
                alertPriority=decision.alert_priority.value,
                educationalOnly=True,
            ),
            priceAlerts=[a.message for a in price_alerts],
            alertId=alert_id,
            lifecycleStatus=lifecycle_status,
        )

    @staticmethod
    def _map_open(dtos: list[OpenTradeSnapshotDto]) -> tuple[OpenTradeSnapshot, ...]:
        return tuple(
            OpenTradeSnapshot(
                symbol=d.symbol.upper(),
                setup_name=d.setup_name,
                entry_price=d.entry_price,
                stop_price=d.stop_price,
                direction=d.direction,
                position_risk_pct=d.position_risk_pct,
            )
            for d in dtos
        )

    @staticmethod
    def _map_closed(
        dtos: list[ClosedTradeSnapshotDto],
    ) -> tuple[ClosedTradeSnapshot, ...]:
        return tuple(
            ClosedTradeSnapshot(
                setup_name=d.setup_name,
                return_pct=d.return_pct,
                symbol=d.symbol.upper() if d.symbol else "",
            )
            for d in dtos
        )

    @staticmethod
    def _map_settings(dto: AlertRiskSettingsDto | None) -> AlertRiskSettings:
        if dto is None:
            return AlertRiskSettings()
        return AlertRiskSettings(
            max_open_positions=dto.max_open_positions,
            max_risk_per_trade_pct=dto.max_risk_per_trade_pct,
            max_total_open_risk_pct=dto.max_total_open_risk_pct,
            consecutive_loss_limit=dto.consecutive_loss_limit,
            rolling_window_trades=dto.rolling_window_trades,
            rolling_drawdown_limit_pct=dto.rolling_drawdown_limit_pct,
            mega_cap_correlation_threshold=dto.mega_cap_correlation_threshold,
            volume_climax_ratio=dto.volume_climax_ratio,
        )

    @staticmethod
    def _parse_regime(value: str | None) -> MarketRegime | None:
        if not value:
            return None
        try:
            return MarketRegime(value.upper())
        except ValueError:
            return None
