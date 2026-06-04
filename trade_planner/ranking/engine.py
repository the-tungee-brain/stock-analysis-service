"""Cross-sectional stock ranking from trend, RS, volume, and setup quality."""

from __future__ import annotations

from trade_planner.config import RankingConfig, TradePlannerConfig
from trade_planner.indicators import (
    normalize_score,
    relative_strength,
    trend_strength_pct,
    volume_expansion_ratio,
)
from trade_planner.models import StockRank
from trade_planner.protocols import Setup
from trade_planner.types import OHLCVBar, StockData


class StockRankingEngine:
    def __init__(self, config: RankingConfig | None = None) -> None:
        self._config = config or TradePlannerConfig().ranking

    def score_symbol(
        self,
        *,
        symbol: str,
        bars: tuple[OHLCVBar, ...] | list[OHLCVBar],
        setups: list[Setup],
        benchmark_bars: tuple[OHLCVBar, ...] | list[OHLCVBar] | None = None,
    ) -> StockRank:
        frozen = tuple(bars) if not isinstance(bars, tuple) else bars
        if not frozen:
            return StockRank(
                symbol=symbol,
                score=0.0,
                trend_strength=0.0,
                relative_strength=0.0,
                volume_expansion=0.0,
                setup_quality=0.0,
                best_setup=None,
            )

        cfg = self._config
        data = StockData.from_bars(symbol, frozen)
        window = data.slice_end(max(cfg.trend_lookback_days, cfg.volume_avg_days) + 1)

        trend = trend_strength_pct(window, cfg.trend_lookback_days) or 0.0
        vol_ratio = volume_expansion_ratio(window, cfg.volume_avg_days) or 1.0

        rs = 0.0
        if benchmark_bars:
            bench = tuple(benchmark_bars) if not isinstance(benchmark_bars, tuple) else benchmark_bars
            if bench:
                bench_data = StockData.from_bars("BENCHMARK", bench)
                rs = (
                    relative_strength(
                        window,
                        bench_data.slice_end(cfg.rs_lookback_days),
                        cfg.rs_lookback_days,
                    )
                    or 0.0
                )

        setup_scores: list[tuple[str, float]] = []
        for setup in setups:
            if setup.is_valid(data):
                setup_scores.append((setup.name, setup.confidence_score(data)))

        setup_quality = max((score for _, score in setup_scores), default=0.0)
        best_setup = max(setup_scores, key=lambda item: item[1])[0] if setup_scores else None

        trend_score = normalize_score(trend, low=-0.05, high=0.15)
        rs_score = normalize_score(rs, low=-0.05, high=0.10)
        vol_score = normalize_score(vol_ratio, low=0.8, high=2.5)
        setup_score = setup_quality

        composite = (
            trend_score * cfg.trend_weight
            + rs_score * cfg.relative_strength_weight
            + vol_score * cfg.volume_weight
            + setup_score * cfg.setup_quality_weight
        )

        return StockRank(
            symbol=symbol,
            score=round(min(100.0, max(0.0, composite)), 2),
            trend_strength=round(trend, 6),
            relative_strength=round(rs, 6),
            volume_expansion=round(vol_ratio, 4),
            setup_quality=round(setup_quality, 2),
            best_setup=best_setup,
        )

    def rank_symbols(
        self,
        symbol_bars: dict[str, tuple[OHLCVBar, ...] | list[OHLCVBar]],
        setups: list[Setup],
        *,
        benchmark_bars: tuple[OHLCVBar, ...] | list[OHLCVBar] | None = None,
    ) -> list[StockRank]:
        ranks = [
            self.score_symbol(
                symbol=symbol,
                bars=bars,
                setups=setups,
                benchmark_bars=benchmark_bars,
            )
            for symbol, bars in symbol_bars.items()
        ]
        return sorted(ranks, key=lambda item: item.score, reverse=True)
