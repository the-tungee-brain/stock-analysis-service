from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.yfinance_analysis_models import (
    AnalystPriceTargets,
    PeriodEstimate,
    RecommendationBreakdown,
    StreetAnalysisSnapshot,
)

logger = logging.getLogger(__name__)

_PERIOD_LABELS = {
    "0q": "Current quarter",
    "+1q": "Next quarter",
    "0y": "Current year",
    "+1y": "Next year",
}


class YFinanceAnalysisBuilder:
    def __init__(self, yfinance_adapter: YFinanceAdapter):
        self.yfinance_adapter = yfinance_adapter

    def build(self, symbol: str) -> StreetAnalysisSnapshot | None:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper:
            return None

        try:
            raw = self.yfinance_adapter.get_street_analysis_raw(symbol_upper)
        except Exception:
            logger.exception("yfinance street analysis failed for %s", symbol_upper)
            return None

        info = self.yfinance_adapter.get_ticker_info(symbol_upper)
        current_price = self._optional_float(info.get("currentPrice") or info.get("regularMarketPrice"))

        price_targets = self._parse_price_targets(raw.get("price_targets"), current_price)
        recommendation = self._parse_recommendations(raw.get("recommendations_summary"))
        next_q_eps = self._parse_period_estimate(
            raw.get("earnings_estimate"), "+1q", year_ago_key="yearAgoEps"
        )
        next_q_revenue = self._parse_period_estimate(
            raw.get("revenue_estimate"), "+1q", year_ago_key="yearAgoRevenue"
        )
        revision_headline = self._revision_headline(raw.get("eps_revisions"), "+1q")

        if not any(
            [
                price_targets,
                recommendation,
                next_q_eps,
                next_q_revenue,
                revision_headline,
            ]
        ):
            return None

        consensus_label = (
            self._consensus_label(recommendation) if recommendation else None
        )

        return StreetAnalysisSnapshot(
            price_targets=price_targets,
            recommendation=recommendation,
            consensus_label=consensus_label,
            next_quarter_eps=next_q_eps,
            next_quarter_revenue=next_q_revenue,
            estimate_revision_headline=revision_headline,
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return parsed

    def _parse_price_targets(
        self, raw: Any, current_price: float | None
    ) -> AnalystPriceTargets | None:
        if not isinstance(raw, dict) or not raw:
            return None

        mean = self._optional_float(raw.get("mean"))
        targets = AnalystPriceTargets(
            current=self._optional_float(raw.get("current")) or current_price,
            low=self._optional_float(raw.get("low")),
            high=self._optional_float(raw.get("high")),
            mean=mean,
            median=self._optional_float(raw.get("median")),
        )
        if targets.current and mean:
            targets.upside_to_mean_pct = (
                (mean - targets.current) / abs(targets.current)
            ) * 100

        if not any([targets.low, targets.high, targets.mean, targets.median]):
            return None
        return targets

    def _parse_recommendations(self, df: Any) -> RecommendationBreakdown | None:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None

        row = df.iloc[0]
        if "period" in df.columns:
            current = df[df["period"].astype(str).str.lower().isin(["0m", "1m", "-1"])]
            if not current.empty:
                row = current.iloc[0]
            elif len(df) > 1:
                row = df.iloc[-1]

        def col(name: str) -> int:
            for key in (name, name.replace("_", "")):
                if key in row.index:
                    value = row[key]
                    if value is not None and not pd.isna(value):
                        return int(value)
            return 0

        breakdown = RecommendationBreakdown(
            strong_buy=col("strongBuy"),
            buy=col("buy"),
            hold=col("hold"),
            sell=col("sell"),
            strong_sell=col("strongSell"),
        )
        total = (
            breakdown.strong_buy
            + breakdown.buy
            + breakdown.hold
            + breakdown.sell
            + breakdown.strong_sell
        )
        return breakdown if total > 0 else None

    @staticmethod
    def _consensus_label(rec: RecommendationBreakdown) -> str:
        total = rec.strong_buy + rec.buy + rec.hold + rec.sell + rec.strong_sell
        if total <= 0:
            return "No consensus"

        bullish = rec.strong_buy + rec.buy
        bearish = rec.sell + rec.strong_sell
        if bullish >= max(bearish * 2, total * 0.55):
            return "Mostly Buy"
        if bearish >= max(bullish * 2, total * 0.4):
            return "Mostly Sell"
        if rec.hold >= total * 0.45:
            return "Hold"
        return "Mixed"

    def _parse_period_estimate(
        self,
        table: Any,
        period_key: str,
        *,
        year_ago_key: str,
    ) -> PeriodEstimate | None:
        row = self._table_row(table, period_key)
        if row is None:
            return None

        avg = self._optional_float(row.get("avg"))
        if avg is None:
            return None

        growth = self._optional_float(row.get("growth"))
        if growth is not None and abs(growth) <= 1.5:
            growth *= 100

        return PeriodEstimate(
            period_key=period_key,
            label=_PERIOD_LABELS.get(period_key, period_key),
            analyst_count=self._optional_int(row.get("numberOfAnalysts")),
            avg=avg,
            low=self._optional_float(row.get("low")),
            high=self._optional_float(row.get("high")),
            growth_pct=growth,
        )

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return None
        return parsed

    @staticmethod
    def _table_row(table: Any, period_key: str) -> dict[str, Any] | None:
        if table is None:
            return None
        if isinstance(table, pd.DataFrame):
            if table.empty or period_key not in table.index.astype(str):
                return None
            row = table.loc[period_key]
            return {str(k): row[k] for k in row.index}
        if isinstance(table, dict) and period_key in table:
            row = table[period_key]
            return row if isinstance(row, dict) else None
        return None

    def _revision_headline(self, table: Any, period_key: str) -> str | None:
        row = self._table_row(table, period_key)
        if row is None:
            return None

        up30 = self._optional_int(row.get("upLast30days"))
        down30 = self._optional_int(row.get("downLast30days"))
        if up30 is None and down30 is None:
            return None

        up30 = up30 or 0
        down30 = down30 or 0
        if up30 == 0 and down30 == 0:
            return "No next-quarter EPS estimate revisions in the last 30 days."
        if up30 > down30:
            return (
                f"Next-quarter EPS estimates revised up {up30}× vs down {down30}× "
                "in the last 30 days."
            )
        if down30 > up30:
            return (
                f"Next-quarter EPS estimates revised down {down30}× vs up {up30}× "
                "in the last 30 days."
            )
        return (
            f"Next-quarter EPS estimates revised up {up30}× and down {down30}× "
            "in the last 30 days."
        )
