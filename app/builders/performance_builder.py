from datetime import timedelta
import pandas as pd
from app.adapters.market.market_data_adapter import MarketDataAdapter
from app.models.company_research_models import PerformanceSnapshot


class PerformanceBuilder:
    def __init__(self, market_data_adapter: MarketDataAdapter):
        self.market_data_adapter = market_data_adapter

    def build(self, symbol: str) -> PerformanceSnapshot:
        closes = self.market_data_adapter.get_daily_closes_1y(symbol=symbol)

        if closes.empty:
            return PerformanceSnapshot(
                oneMonth="N/A",
                threeMonth="N/A",
                oneYear="N/A",
                trendLabel="Not enough data to show a trend yet.",
                volatilityNote="Price history is too short for a useful view.",
            )

        r_1m = self._compute_period_return(closes=closes, days=30)
        r_3m = self._compute_period_return(closes=closes, days=90)
        r_1y = self._compute_period_return(closes=closes, days=365)

        def fmt(r: float | None) -> str:
            if r is None:
                return "N/A"
            sign = "+" if r >= 0 else ""
            return f"{sign}{r:.1f}%"

        one_month = fmt(r_1m)
        three_month = fmt(r_3m)
        one_year = fmt(r_1y)

        trend_label = self._trend_label(r_1m, r_3m, r_1y)
        volatility_note = (
            "The stock can move sharply in the short term, especially around "
            "earnings and macro news."
        )

        return PerformanceSnapshot(
            oneMonth=one_month,
            threeMonth=three_month,
            oneYear=one_year,
            trendLabel=trend_label,
            volatilityNote=volatility_note,
        )

    def _compute_period_return(self, closes: pd.Series, days: int) -> float | None:
        if closes.empty:
            return None
        end_price = float(closes.iloc[-1])
        cutoff_date = closes.index[-1] - timedelta(days=days)
        past = closes[closes.index <= cutoff_date]
        if past.empty:
            return None
        start_price = float(past.iloc[-1])
        if start_price == 0:
            return None
        return (end_price / start_price - 1.0) * 100.0

    def _trend_label(
        self,
        r_1m: float | None,
        r_3m: float | None,
        r_1y: float | None,
    ) -> str:
        r1 = r_1y or 0
        r3 = r_3m or 0
        r0 = r_1m or 0

        if r1 > 0 and r3 > 0:
            return "Uptrend over the past year, with recent gains."
        if r1 < 0 and r3 < 0:
            return "Downtrend over the past year, with recent weakness."
        if r0 > 0 and r1 > 0:
            return "Mostly positive, but with some pullbacks along the way."
        if r0 < 0 and r1 > 0:
            return "Long‑term uptrend with a recent pullback."
        return "Mixed performance with ups and downs over the past year."
