"""Scan symbols for valid Momentum Breakout setups (educational, no execution)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

from data.benchmarks import BENCHMARK_SYMBOL
from data.loader import load_symbol
from data.store import raw_exists
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store
from trade_planner.alerts.risk_gate import AlertRiskGate
from trade_planner.alerts.risk_models import AlertRiskContext, AlertRiskSettings
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import TradePlannerConfig
from trade_planner.research.data import align_benchmark_to_stock, ohlcv_bars_from_dataframe
from trade_planner.research.features import capture_momentum_feature_snapshot
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import OHLCVBar, StockData

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.momentum_breakout_scan_models import (
    DEFAULT_MAX_STOP_DISTANCE_PCT,
    DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
    DEFAULT_MIN_HISTORICAL_TRADES,
    MomentumBreakoutScanCandidateDto,
    MomentumBreakoutScanResponse,
    MomentumBreakoutUniverseResponse,
    UNIVERSE_SOURCE_DESCRIPTION,
)

logger = logging.getLogger(__name__)

TOP_CANDIDATES_LIMIT = 20


def _max_universe() -> int:
    return int(os.environ.get("MB_SCAN_MAX_UNIVERSE", "500"))


def _worker_count() -> int:
    return int(os.environ.get("MB_SCAN_WORKERS", "8"))


def compute_stop_distance_pct(entry_price: float, stop_price: float) -> float:
    """Percent distance from entry to stop (long setups)."""
    if entry_price <= 0:
        return 0.0
    return round(abs(entry_price - stop_price) / entry_price * 100.0, 4)


def is_tradable_candidate(
    candidate: "_ScanCandidate",
    *,
    min_historical_profit_factor: float = DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
    min_historical_trades: int = DEFAULT_MIN_HISTORICAL_TRADES,
    max_stop_distance_pct: float = DEFAULT_MAX_STOP_DISTANCE_PCT,
) -> bool:
    if not candidate.risk_gate.allowed:
        return False
    profit_factor = candidate.historical_profit_factor
    if profit_factor is None or profit_factor < min_historical_profit_factor:
        return False
    total_trades = candidate.historical_total_trades
    if total_trades is None or total_trades < min_historical_trades:
        return False
    if candidate.stop_distance_pct > max_stop_distance_pct:
        return False
    return True


def _parse_symbols_param(symbols: str | None) -> list[str] | None:
    if not symbols or not symbols.strip():
        return None
    seen: set[str] = set()
    ordered: list[str] = []
    for part in symbols.split(","):
        sym = part.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return ordered


def _ranking_universe_snapshot_symbols() -> tuple[str, list[str]]:
    """Active ranking snapshot members that passed liquidity filters (alphabetical)."""
    cfg = default_config()
    store = open_store(cfg)
    snapshot_id = store.active_snapshot_id()
    if not snapshot_id:
        raise LookupError("No active ranking universe snapshot")
    universe = store.load_universe_symbols(snapshot_id)
    if not universe:
        raise LookupError("No active ranking universe")
    return snapshot_id, universe


def build_production_scan_universe(
    *,
    max_symbols: int | None = None,
    sample_size: int = 50,
) -> MomentumBreakoutUniverseResponse:
    """Symbols the production scanner evaluates when `symbols` query param is omitted."""
    cap = max(1, max_symbols if max_symbols is not None else _max_universe())
    _snapshot_id, universe = _ranking_universe_snapshot_symbols()
    with_data = [s.strip().upper() for s in universe if raw_exists(s.strip().upper())]
    scanned = with_data[:cap]
    total_available = len(with_data)
    return MomentumBreakoutUniverseResponse(
        totalAvailableSymbols=total_available,
        scanCap=cap,
        symbolsScanned=len(scanned),
        excludedSymbols=max(0, total_available - len(scanned)),
        universeSource=UNIVERSE_SOURCE_DESCRIPTION,
        sampleSymbols=scanned[: max(0, sample_size)],
    )


def _load_ranking_universe(*, max_symbols: int) -> list[str]:
    _snapshot_id, universe = _ranking_universe_snapshot_symbols()
    with_data = [s.strip().upper() for s in universe if raw_exists(s.strip().upper())]
    return with_data[: max(1, max_symbols)]


@dataclass(frozen=True, slots=True)
class _ScanCandidate:
    symbol: str
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float
    historical_win_rate: float | None
    historical_profit_factor: float | None
    historical_total_trades: int | None
    setup_score: float
    stop_distance_pct: float
    volume_ratio: float | None
    rs_percentile: float | None
    market_regime: str | None
    risk_gate: AlertRiskGateResultDto


def _candidate_sort_key(candidate: _ScanCandidate) -> tuple[float, float]:
    profit_factor = (
        candidate.historical_profit_factor
        if candidate.historical_profit_factor is not None
        else -1.0
    )
    return (-candidate.setup_score, -profit_factor)


def _risk_gate_dto(decision) -> AlertRiskGateResultDto:
    return AlertRiskGateResultDto(
        allowed=decision.allowed,
        action=decision.action.value,
        reasons=list(decision.reasons),
        recommendedPositionRiskPct=decision.recommended_position_risk_pct,
        maxNotionalUsd=decision.max_shares_or_dollars,
        alertPriority=decision.alert_priority.value,
        educationalOnly=True,
    )


class MomentumBreakoutScannerService:
    """Find symbols that satisfy Momentum Breakout rules on the latest bar."""

    def __init__(
        self,
        *,
        setup: MomentumBreakoutSetup | None = None,
        risk_gate: AlertRiskGate | None = None,
    ) -> None:
        self._config = TradePlannerConfig()
        self._setup = setup or MomentumBreakoutSetup(self._config.momentum)
        self._risk_gate = risk_gate or AlertRiskGate()
        self._benchmark_bars: tuple[OHLCVBar, ...] | None = None

    def _benchmark_bars_cached(self) -> tuple[OHLCVBar, ...]:
        if self._benchmark_bars is None:
            bench_df = load_symbol(BENCHMARK_SYMBOL)
            self._benchmark_bars = ohlcv_bars_from_dataframe(bench_df)
        return self._benchmark_bars

    def resolve_symbol_list(self, symbols: str | None) -> list[str]:
        explicit = _parse_symbols_param(symbols)
        if explicit is not None:
            return explicit
        return _load_ranking_universe(max_symbols=_max_universe())

    def describe_production_universe(
        self,
        *,
        sample_size: int = 50,
    ) -> MomentumBreakoutUniverseResponse:
        return build_production_scan_universe(sample_size=sample_size)

    def evaluate_symbol(self, symbol: str) -> _ScanCandidate | None:
        sym = symbol.strip().upper()
        if not sym:
            return None
        try:
            stock_df = load_symbol(sym)
        except FileNotFoundError:
            return None

        stock_bars = ohlcv_bars_from_dataframe(stock_df)
        bench_bars = align_benchmark_to_stock(
            stock_bars, self._benchmark_bars_cached()
        )
        data = StockData.from_bars(sym, stock_bars, benchmark_bars=bench_bars)

        if not self._setup.is_valid(data):
            return None

        plan = self._setup.build_plan(data)
        if plan is None:
            return None

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
        regime = snapshot.market_regime.value

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

        return _ScanCandidate(
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
            market_regime=regime,
            risk_gate=_risk_gate_dto(decision),
        )

    def scan(
        self,
        *,
        symbols: str | None = None,
        limit: int = 50,
        tradable_only: bool = False,
        min_historical_profit_factor: float = DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
        min_historical_trades: int = DEFAULT_MIN_HISTORICAL_TRADES,
        max_stop_distance_pct: float = DEFAULT_MAX_STOP_DISTANCE_PCT,
    ) -> MomentumBreakoutScanResponse:
        symbol_list = self.resolve_symbol_list(symbols)
        valid_candidates = self._collect_candidates(symbol_list)
        valid_candidates.sort(key=_candidate_sort_key)

        tradable_candidates = [
            candidate
            for candidate in valid_candidates
            if is_tradable_candidate(
                candidate,
                min_historical_profit_factor=min_historical_profit_factor,
                min_historical_trades=min_historical_trades,
                max_stop_distance_pct=max_stop_distance_pct,
            )
        ]

        pool = tradable_candidates if tradable_only else valid_candidates
        capped = pool[: max(1, limit)]

        return MomentumBreakoutScanResponse(
            scanTime=datetime.now(timezone.utc).isoformat(),
            totalSymbolsScanned=len(symbol_list),
            validSetupsFound=len(valid_candidates),
            tradableCandidatesFound=len(tradable_candidates),
            blockedCandidatesCount=len(valid_candidates) - len(tradable_candidates),
            candidatesFound=len(capped),
            candidates=[self._to_dto(c) for c in capped],
        )

    def top_candidates(
        self,
        *,
        tradable_only: bool = False,
        min_historical_profit_factor: float = DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR,
        min_historical_trades: int = DEFAULT_MIN_HISTORICAL_TRADES,
        max_stop_distance_pct: float = DEFAULT_MAX_STOP_DISTANCE_PCT,
    ) -> MomentumBreakoutScanResponse:
        return self.scan(
            limit=TOP_CANDIDATES_LIMIT,
            tradable_only=tradable_only,
            min_historical_profit_factor=min_historical_profit_factor,
            min_historical_trades=min_historical_trades,
            max_stop_distance_pct=max_stop_distance_pct,
        )

    def _collect_candidates(self, symbol_list: list[str]) -> list[_ScanCandidate]:
        if not symbol_list:
            return []

        self._benchmark_bars_cached()
        workers = max(1, min(_worker_count(), 16, len(symbol_list)))
        found: list[_ScanCandidate] = []

        if workers == 1:
            for sym in symbol_list:
                candidate = self.evaluate_symbol(sym)
                if candidate is not None:
                    found.append(candidate)
            return found

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.evaluate_symbol, sym): sym for sym in symbol_list}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    candidate = future.result()
                except Exception:
                    logger.exception("Momentum scan failed for %s", sym)
                    continue
                if candidate is not None:
                    found.append(candidate)
        return found

    @staticmethod
    def _to_dto(candidate: _ScanCandidate) -> MomentumBreakoutScanCandidateDto:
        return MomentumBreakoutScanCandidateDto(
            symbol=candidate.symbol,
            entryPrice=candidate.entry_price,
            stopPrice=candidate.stop_price,
            targetPrice=candidate.target_price,
            riskReward=candidate.risk_reward,
            historicalWinRate=candidate.historical_win_rate,
            historicalProfitFactor=candidate.historical_profit_factor,
            historicalTotalTrades=candidate.historical_total_trades,
            setupScore=candidate.setup_score,
            stopDistancePct=candidate.stop_distance_pct,
            volumeRatio=candidate.volume_ratio,
            rsPercentile=candidate.rs_percentile,
            marketRegime=candidate.market_regime,
            riskGate=candidate.risk_gate,
        )
