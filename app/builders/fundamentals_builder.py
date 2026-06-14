from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.company_research_models import FundamentalMetric
from app.utils.dividend_yield import dividend_yield_pct_or_none


class FundamentalsBuilder:
    def __init__(self, market_data_adapter: YFinanceAdapter):
        self.market_data_adapter = market_data_adapter

    def build(self, symbol: str) -> list[FundamentalMetric]:
        info = self.market_data_adapter.get_ticker_info(symbol=symbol)
        if not info:
            return []

        metrics: list[FundamentalMetric] = []
        asset_type = self._asset_type(info)

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
            self._fmt_dividend_yield(
                info.get("dividendYield"),
                asset_type=asset_type,
                symbol=symbol,
            ),
            "Annual dividend as a share of the stock price. Relevant for income-focused investors.",
        )
        add(
            "Annual dividend per share",
            self._fmt_dollar(info.get("dividendRate")),
            "Trailing annual dividend per share before reinvestment.",
        )
        add(
            "Payout ratio",
            self._fmt_payout_ratio(self._resolve_payout_ratio(info)),
            "Share of earnings paid out as dividends. Lower ratios leave more room for reinvestment or downturns.",
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

    def build_etf_metrics(self, symbol: str) -> dict[str, str | None]:
        info = self.market_data_adapter.get_ticker_info(symbol=symbol)
        if not info:
            return {"dividend_yield": None, "expense_ratio": None}
        return {
            "dividend_yield": self._fmt_dividend_yield(
                info.get("dividendYield"),
                asset_type="ETF",
                symbol=symbol,
            ),
            "expense_ratio": self._fmt_expense_ratio(info),
        }

    @staticmethod
    def _resolve_payout_ratio(info: dict) -> float | None:
        raw = info.get("payoutRatio")
        if isinstance(raw, (int, float)):
            return float(raw)

        dividend_rate = info.get("trailingAnnualDividendRate") or info.get("dividendRate")
        trailing_eps = info.get("trailingEps")
        if (
            isinstance(dividend_rate, (int, float))
            and isinstance(trailing_eps, (int, float))
            and trailing_eps > 0
        ):
            return float(dividend_rate) / float(trailing_eps)

        return None

    @staticmethod
    def _fmt_payout_ratio(value: float | None) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        pct = value * 100 if abs(value) <= 1.5 else value
        return f"{pct:.1f}%"

    @staticmethod
    def _fmt_dividend_yield(
        value: float | None,
        *,
        asset_type: str | None = None,
        symbol: str | None = None,
    ) -> str | None:
        pct = dividend_yield_pct_or_none(
            value,
            asset_type=asset_type,
            source="yfinance.info.dividendYield",
            symbol=symbol,
        )
        if pct is None:
            return None
        return f"{pct:.2f}%"

    @staticmethod
    def _asset_type(info: dict) -> str:
        quote_type = str(info.get("quoteType") or "").upper()
        if quote_type == "ETF":
            return "ETF"
        legal_type = str(info.get("legalType") or "").lower()
        if "exchange traded fund" in legal_type:
            return "ETF"
        return "STOCK"

    @staticmethod
    def _fmt_expense_ratio(info: dict) -> str | None:
        annual = info.get("annualReportExpenseRatio")
        if isinstance(annual, (int, float)) and annual > 0:
            return f"{abs(annual) * 100:.2f}%"

        for key in ("netExpenseRatio", "expenseRatio"):
            value = info.get(key)
            if value is None or not isinstance(value, (int, float)) or value <= 0:
                continue
            abs_value = abs(value)
            if abs_value < 0.01:
                return f"{abs_value * 100:.2f}%"
            return f"{abs_value:.2f}%"
        return None

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
