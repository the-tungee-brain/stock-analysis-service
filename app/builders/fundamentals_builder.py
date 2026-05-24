from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.company_research_models import FundamentalMetric


class FundamentalsBuilder:
    def __init__(self, market_data_adapter: YFinanceAdapter):
        self.market_data_adapter = market_data_adapter

    def build(self, symbol: str) -> list[FundamentalMetric]:
        info = self.market_data_adapter.get_ticker_info(symbol=symbol)
        if not info:
            return []

        metrics: list[FundamentalMetric] = []

        def add(label: str, value: str | None, note: str) -> None:
            if value is not None:
                metrics.append(FundamentalMetric(label=label, value=value, note=note))

        add(
            "P/E (trailing)",
            self._fmt_multiple(info.get("trailingPE")),
            "Price divided by earnings over the last 12 months. Higher values often reflect stronger growth expectations.",
        )
        add(
            "P/E (forward)",
            self._fmt_multiple(info.get("forwardPE")),
            "Price divided by expected earnings over the next 12 months. Compare to trailing P/E to see if growth is priced in.",
        )
        add(
            "Price / book",
            self._fmt_multiple(info.get("priceToBook")),
            "Price relative to accounting book value. Below 1.0 can mean undervalued — or troubled.",
        )
        add(
            "Profit margin",
            self._fmt_pct(info.get("profitMargins")),
            "Net income as a share of revenue. Shows how much profit the company keeps from each dollar of sales.",
        )
        add(
            "Operating margin",
            self._fmt_pct(info.get("operatingMargins")),
            "Operating income as a share of revenue. Measures core business profitability before interest and taxes.",
        )
        add(
            "Gross margin",
            self._fmt_pct(info.get("grossMargins")),
            "Revenue minus cost of goods sold, as a share of revenue. Higher margins often signal pricing power.",
        )
        add(
            "Revenue growth",
            self._fmt_pct(info.get("revenueGrowth")),
            "Year-over-year revenue change. Positive growth suggests expanding demand or market share.",
        )
        add(
            "Earnings growth",
            self._fmt_pct(info.get("earningsGrowth")),
            "Year-over-year earnings change. Watch whether growth is driven by revenue or cost cuts.",
        )
        add(
            "Return on equity",
            self._fmt_pct(info.get("returnOnEquity")),
            "Net income relative to shareholder equity. Measures how efficiently the company uses investor capital.",
        )
        add(
            "Return on assets",
            self._fmt_pct(info.get("returnOnAssets")),
            "Net income relative to total assets. Useful for comparing capital-heavy vs. asset-light businesses.",
        )
        add(
            "Debt / equity",
            self._fmt_ratio(info.get("debtToEquity")),
            "Total debt relative to shareholder equity. Higher leverage amplifies both gains and losses.",
        )
        add(
            "Current ratio",
            self._fmt_ratio(info.get("currentRatio")),
            "Current assets divided by current liabilities. Above 1.0 suggests the company can cover short-term obligations.",
        )
        add(
            "Free cash flow",
            self._fmt_large_number(info.get("freeCashflow")),
            "Cash generated after operating expenses and capital spending. Positive FCF supports dividends, buybacks, and growth.",
        )
        add(
            "Dividend yield",
            self._fmt_pct(info.get("dividendYield")),
            "Annual dividend as a share of the stock price. Relevant for income-focused investors.",
        )
        add(
            "Beta",
            self._fmt_ratio(info.get("beta")),
            "Sensitivity to market moves. Beta above 1.0 means the stock tends to move more than the overall market.",
        )
        add(
            "EPS (trailing)",
            self._fmt_dollar(info.get("trailingEps")),
            "Earnings per share over the last 12 months. The denominator in the trailing P/E ratio.",
        )
        add(
            "EPS (forward)",
            self._fmt_dollar(info.get("forwardEps")),
            "Expected earnings per share over the next 12 months. The denominator in the forward P/E ratio.",
        )

        return metrics

    def _fmt_multiple(self, value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        return f"{value:.1f}x"

    def _fmt_pct(self, value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        return f"{value * 100:.1f}%"

    def _fmt_ratio(self, value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        return f"{value:.2f}"

    def _fmt_dollar(self, value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        return f"${value:.2f}"

    def _fmt_large_number(self, value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000_000:.1f}T"
        if abs_val >= 1_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000:.1f}B"
        if abs_val >= 1_000_000:
            return f"{sign}${abs_val / 1_000_000:.1f}M"
        return f"{sign}${abs_val:,.0f}"
