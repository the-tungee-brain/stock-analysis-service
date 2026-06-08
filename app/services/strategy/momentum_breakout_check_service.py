"""Single-symbol Momentum Breakout setup check for manual stock lookup."""

from __future__ import annotations

from data.benchmarks import BENCHMARK_SYMBOL
from data.loader import load_symbol
from trade_planner.alerts.risk_gate import AlertRiskGate
from trade_planner.alerts.risk_models import AlertRiskContext, AlertRiskSettings
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import TradePlannerConfig
from trade_planner.research.data import align_stock_and_benchmark, ohlcv_bars_from_dataframe
from trade_planner.research.features import capture_momentum_feature_snapshot
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.setups.momentum_breakout_diagnostics import diagnose_momentum_breakout_setup
from trade_planner.types import StockData

from app.core.momentum_breakout_feature_flags import mb_alert_creation_enabled
from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_check_models import MomentumBreakoutCheckResponse
from app.models.momentum_breakout_scan_models import (
    DEFAULT_MAX_STOP_DISTANCE_PCT,
    DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
    DEFAULT_MIN_HISTORICAL_TRADES,
)
from app.services.strategy.momentum_breakout_scanner_service import (
    _ScanCandidate,
    _risk_gate_dto,
    compute_stop_distance_pct,
    is_tradable_candidate,
)


def _rejection_reasons(candidate: _ScanCandidate) -> list[str]:
    reasons: list[str] = []
    if not candidate.risk_gate.allowed:
        reasons.extend(candidate.risk_gate.reasons)
    profit_factor = candidate.historical_profit_factor
    if profit_factor is None or profit_factor < DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR:
        reasons.append(
            "Historical performance on this pattern is below our minimum threshold."
        )
    total_trades = candidate.historical_total_trades
    if total_trades is None or total_trades < DEFAULT_MIN_HISTORICAL_TRADES:
        reasons.append(
            f"Not enough historical examples for this stock "
            f"({total_trades or 0} studied, {DEFAULT_MIN_HISTORICAL_TRADES} required)."
        )
    if candidate.stop_distance_pct > DEFAULT_MAX_STOP_DISTANCE_PCT:
        reasons.append(
            f"Stop distance is too wide ({candidate.stop_distance_pct:.1f}% vs "
            f"{DEFAULT_MAX_STOP_DISTANCE_PCT:.0f}% maximum)."
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


class MomentumBreakoutCheckService:
    def __init__(
        self,
        *,
        setup: MomentumBreakoutSetup | None = None,
        risk_gate: AlertRiskGate | None = None,
    ) -> None:
        self._config = TradePlannerConfig()
        self._setup = setup or MomentumBreakoutSetup(self._config.momentum)
        self._risk_gate = risk_gate or AlertRiskGate()

    def check(self, symbol: str) -> MomentumBreakoutCheckResponse:
        sym = symbol.strip().upper()
        if not sym:
            raise ValueError("symbol is required")

        try:
            stock_df = load_symbol(sym)
            bench_df = load_symbol(BENCHMARK_SYMBOL)
        except FileNotFoundError:
            return MomentumBreakoutCheckResponse(
                symbol=sym,
                status="DATA_UNAVAILABLE",
                verdictTitle=f"We do not have enough data for {sym}",
                verdictMessage=(
                    "We could not load daily price history for this symbol. "
                    "Try a US-listed ticker we cover, or check back after data updates."
                ),
                canTrackBreakoutPlan=False,
            )

        stock_bars, bench_bars = align_stock_and_benchmark(
            ohlcv_bars_from_dataframe(stock_df),
            ohlcv_bars_from_dataframe(bench_df),
        )
        if not stock_bars or len(stock_bars) != len(bench_bars):
            return MomentumBreakoutCheckResponse(
                symbol=sym,
                status="DATA_UNAVAILABLE",
                verdictTitle=f"We do not have enough data for {sym}",
                verdictMessage=(
                    "We could not align this symbol's daily price history with "
                    "the benchmark used by Momentum Breakout."
                ),
                canTrackBreakoutPlan=False,
            )
        data = StockData.from_bars(sym, stock_bars, benchmark_bars=bench_bars)

        diagnostics = diagnose_momentum_breakout_setup(data, self._setup)
        if not diagnostics.setup_valid:
            return MomentumBreakoutCheckResponse(
                symbol=sym,
                status="NO_BREAKOUT_SETUP",
                verdictTitle=f"{sym} does not have a Momentum Breakout setup today",
                verdictMessage=(
                    "This stock does not meet all Momentum Breakout criteria on the latest bar. "
                    "You may still explore a separate custom educational plan."
                ),
                failedSetupRules=diagnostics.failed_setup_rules,
                canTrackBreakoutPlan=False,
            )

        plan = self._setup.build_plan(data)
        if plan is None:
            return MomentumBreakoutCheckResponse(
                symbol=sym,
                status="NO_BREAKOUT_SETUP",
                verdictTitle=f"{sym} does not have a Momentum Breakout setup today",
                verdictMessage=(
                    "A breakout pattern was not confirmed on the latest bar."
                ),
                failedSetupRules=diagnostics.failed_setup_rules,
                canTrackBreakoutPlan=False,
            )

        engine = BacktestEngine(self._config.backtest, store=None)
        result = engine.run(
            self._setup,
            stock_bars,
            symbol=sym,
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

        context = AlertRiskContext(
            candidate_plan=plan,
            open_trades=(),
            recent_closed=(),
            current_symbol=sym,
            market_regime=snapshot.market_regime,
            volume_ratio=snapshot.volume_ratio,
            settings=AlertRiskSettings(),
        )
        decision = self._risk_gate.evaluate(context)
        risk_dto = _risk_gate_dto(decision)

        candidate = _ScanCandidate(
            symbol=sym,
            entry_price=plan.entry_price,
            stop_price=plan.stop_price,
            target_price=plan.target_price,
            risk_reward=plan.risk_reward,
            historical_win_rate=stats.win_rate if stats.total_trades else None,
            historical_profit_factor=stats.profit_factor if stats.total_trades else None,
            historical_total_trades=stats.total_trades or None,
            setup_score=plan.confidence_score,
            stop_distance_pct=compute_stop_distance_pct(
                plan.entry_price, plan.stop_price
            ),
            volume_ratio=snapshot.volume_ratio,
            rs_percentile=snapshot.rs_percentile,
            market_regime=snapshot.market_regime.value,
            risk_gate=risk_dto,
        )

        common_fields = dict(
            symbol=sym,
            entryPrice=plan.entry_price,
            stopPrice=plan.stop_price,
            targetPrice=plan.target_price,
            stopDistancePct=candidate.stop_distance_pct,
            historicalWinRate=candidate.historical_win_rate,
            historicalProfitFactor=candidate.historical_profit_factor,
            historicalTotalTrades=candidate.historical_total_trades,
            riskGate=risk_dto,
        )

        if is_tradable_candidate(candidate):
            can_track = mb_alert_creation_enabled() and risk_dto.allowed
            return MomentumBreakoutCheckResponse(
                status="TRADABLE_BREAKOUT",
                verdictTitle=f"{sym} has a breakout plan ready",
                verdictMessage=(
                    "This stock passes Momentum Breakout setup, quality, and safety checks. "
                    "You can track it as an educational plan on your watchlist."
                ),
                canTrackBreakoutPlan=can_track,
                **common_fields,
            )

        rejection_reasons = _rejection_reasons(candidate)
        can_track = mb_alert_creation_enabled() and risk_dto.allowed
        return MomentumBreakoutCheckResponse(
            status="REJECTED_BREAKOUT",
            verdictTitle=f"{sym} showed a breakout pattern, but we rejected it",
            verdictMessage=(
                "The stock met the breakout pattern, but it did not pass our quality "
                "or safety requirements for tracking."
            ),
            rejectionReasons=rejection_reasons,
            canTrackBreakoutPlan=can_track,
            **common_fields,
        )
