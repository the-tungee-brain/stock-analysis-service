from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.performance_builder import PerformanceBuilder
from app.models.intelligence_models import PeerComparison, PeerMetric


class PeerComparisonService:
    MAX_PEERS = 6

    def __init__(
        self,
        yfinance_adapter: YFinanceAdapter,
        performance_builder: PerformanceBuilder,
    ):
        self.yfinance_adapter = yfinance_adapter
        self.performance_builder = performance_builder

    def compare(self, symbol: str, peers: list[str]) -> PeerComparison | None:
        symbol_upper = symbol.strip().upper()
        peer_symbols = [
            peer.upper()
            for peer in peers
            if peer and peer.upper() != symbol_upper
        ][: self.MAX_PEERS]

        if not peer_symbols:
            return None

        target_metrics = self._load_peer_metrics(symbol_upper)
        peer_metrics: list[PeerMetric] = []

        with ThreadPoolExecutor(max_workers=min(len(peer_symbols), 4)) as executor:
            futures = {
                executor.submit(
                    self._load_peer_metrics, peer, include_performance=False
                ): peer
                for peer in peer_symbols
            }
            for future in as_completed(futures):
                peer = futures[future]
                try:
                    metrics = future.result()
                    if metrics is not None:
                        peer_metrics.append(metrics)
                except Exception:
                    continue

        if not peer_metrics:
            return PeerComparison(
                target_symbol=symbol_upper,
                target_one_year_return=target_metrics.one_year_return
                if target_metrics
                else None,
                target_pe_trailing=target_metrics.pe_trailing if target_metrics else None,
            )

        peer_metrics.sort(key=lambda item: item.symbol)
        summary = self._build_summary(
            target=target_metrics,
            peers=peer_metrics,
            symbol=symbol_upper,
        )

        return PeerComparison(
            target_symbol=symbol_upper,
            target_one_year_return=target_metrics.one_year_return
            if target_metrics
            else None,
            target_pe_trailing=target_metrics.pe_trailing if target_metrics else None,
            peers=peer_metrics,
            summary=summary,
        )

    def _load_peer_metrics(
        self, symbol: str, *, include_performance: bool = True
    ) -> PeerMetric | None:
        info = self.yfinance_adapter.get_ticker_info(symbol=symbol)
        if not info:
            return None

        one_year_return = None
        if include_performance:
            performance = self.performance_builder.build(symbol=symbol)
            one_year_return = performance.oneYear

        pe = info.get("trailingPE")
        pe_str = f"{pe:.1f}x" if isinstance(pe, (int, float)) else None

        return PeerMetric(
            symbol=symbol,
            name=info.get("shortName") or info.get("longName"),
            one_year_return=one_year_return,
            pe_trailing=pe_str,
            sector=info.get("sector"),
        )

    @staticmethod
    def _build_summary(
        *,
        target: PeerMetric | None,
        peers: list[PeerMetric],
        symbol: str,
    ) -> str | None:
        if target is None or not peers:
            return None

        target_return = PeerComparisonService._parse_return(target.one_year_return)
        peer_returns = [
            PeerComparisonService._parse_return(peer.one_year_return)
            for peer in peers
            if peer.one_year_return
        ]
        peer_returns = [value for value in peer_returns if value is not None]

        parts: list[str] = []
        if target_return is not None and peer_returns:
            median = sorted(peer_returns)[len(peer_returns) // 2]
            diff = target_return - median
            if diff > 5:
                parts.append(
                    f"{symbol} outperformed peer median 1Y return by {diff:.1f}pp."
                )
            elif diff < -5:
                parts.append(
                    f"{symbol} underperformed peer median 1Y return by {abs(diff):.1f}pp."
                )
            else:
                parts.append(f"{symbol} is in line with peer 1Y returns.")

        target_pe = PeerComparisonService._parse_pe(target.pe_trailing)
        peer_pes = [
            PeerComparisonService._parse_pe(peer.pe_trailing) for peer in peers
        ]
        peer_pes = [value for value in peer_pes if value is not None]
        if target_pe is not None and peer_pes:
            median_pe = sorted(peer_pes)[len(peer_pes) // 2]
            if target_pe > median_pe * 1.15:
                parts.append(
                    f"Trades at a premium P/E ({target.pe_trailing}) vs peer median "
                    f"({median_pe:.1f}x)."
                )
            elif target_pe < median_pe * 0.85:
                parts.append(
                    f"Trades at a discount P/E ({target.pe_trailing}) vs peer median "
                    f"({median_pe:.1f}x)."
                )

        return " ".join(parts) if parts else None

    @staticmethod
    def _parse_return(value: str | None) -> float | None:
        if not value or value == "N/A":
            return None
        cleaned = value.strip().replace("%", "").replace("+", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _parse_pe(value: str | None) -> float | None:
        if not value:
            return None
        cleaned = value.strip().replace("x", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
