from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.company_research_models import (
    FinancialStatementsSnapshot,
    FundamentalMetric,
)

KEY_METRIC_NOTES: dict[str, str] = {
    "Revenue growth": (
        "Year-over-year revenue change on the latest filed annual period, "
        "or market data when statements are unavailable."
    ),
    "Gross margin": "Gross profit as a share of revenue on the latest filed annual period.",
    "Net margin": "Net income as a share of revenue on the latest filed annual period.",
    "Free cash flow": "Free cash flow on the latest filed annual period, or trailing market data.",
    "Debt / equity": "Total debt relative to shareholder equity (ratio).",
    "Current ratio": "Current assets divided by current liabilities.",
}


@dataclass(frozen=True)
class CanonicalFinancialMetrics:
    revenue_growth_yoy: float | None = None
    gross_margin_pct: float | None = None
    net_margin_pct: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow_latest: float | None = None
    free_cash_flow_yoy_pct: float | None = None
    return_on_equity_pct: float | None = None
    payout_ratio_pct: float | None = None
    fcf_dividend_coverage: float | None = None

    def format_revenue_growth(self) -> str | None:
        if self.revenue_growth_yoy is None:
            return None
        return f"{self.revenue_growth_yoy:.1f}%"

    def format_gross_margin(self) -> str | None:
        if self.gross_margin_pct is None:
            return None
        return f"{self.gross_margin_pct:.1f}%"

    def format_net_margin(self) -> str | None:
        if self.net_margin_pct is None:
            return None
        return f"{self.net_margin_pct:.1f}%"

    def format_debt_equity(self) -> str | None:
        if self.debt_to_equity is None:
            return None
        ratio = self.debt_to_equity
        if ratio >= 10:
            return f"{ratio:.1f}x"
        return f"{ratio:.2f}x"

    def format_current_ratio(self) -> str | None:
        if self.current_ratio is None:
            return None
        return f"{self.current_ratio:.2f}"

    def format_free_cash_flow(self) -> str | None:
        if self.free_cash_flow_latest is None:
            return None
        return _fmt_compact_currency(self.free_cash_flow_latest)

    def to_key_metrics(self) -> list[FundamentalMetric]:
        rows: list[tuple[str, str | None]] = [
            ("Revenue growth", self.format_revenue_growth()),
            ("Gross margin", self.format_gross_margin()),
            ("Net margin", self.format_net_margin()),
            ("Free cash flow", self.format_free_cash_flow()),
            ("Debt / equity", self.format_debt_equity()),
            ("Current ratio", self.format_current_ratio()),
        ]
        metrics: list[FundamentalMetric] = []
        for label, value in rows:
            if value is None:
                continue
            metrics.append(
                FundamentalMetric(
                    label=label,
                    value=value,
                    note=KEY_METRIC_NOTES.get(label),
                )
            )
        return metrics


def build_canonical_metrics(
    *,
    info: dict[str, Any],
    snapshot: FinancialStatementsSnapshot | None,
) -> CanonicalFinancialMetrics:
    periods = snapshot.periods if snapshot else []
    latest = periods[0] if periods else None
    prior = periods[1] if len(periods) > 1 else None

    revenue = _line_values(snapshot, "Total revenue")
    net_income = _line_values(snapshot, "Net income")
    gross_profit = _line_values(snapshot, "Gross profit")
    fcf = _line_values(snapshot, "Free cash flow")
    dividends = _line_values(snapshot, "Dividends paid")

    rev_growth = _yoy(revenue, latest, prior)
    if rev_growth is None:
        rev_growth = _normalize_pct(info.get("revenueGrowth"))

    gross_margin = _margin_pct(gross_profit, revenue, latest)
    if gross_margin is None:
        gross_margin = _normalize_pct(info.get("grossMargins"))

    net_margin = _margin_pct(net_income, revenue, latest)
    if net_margin is None:
        net_margin = _normalize_pct(info.get("profitMargins"))

    fcf_latest = fcf.get(latest) if latest else None
    if fcf_latest is None:
        raw_fcf = info.get("freeCashflow")
        if isinstance(raw_fcf, (int, float)):
            fcf_latest = float(raw_fcf)

    payout = _normalize_pct(_resolve_payout_ratio(info))
    coverage: float | None = None
    if latest and fcf_latest is not None and fcf_latest > 0:
        div = dividends.get(latest)
        if div is not None:
            paid = abs(div)
            if paid > 0:
                coverage = fcf_latest / paid

    return CanonicalFinancialMetrics(
        revenue_growth_yoy=rev_growth,
        gross_margin_pct=gross_margin,
        net_margin_pct=net_margin,
        debt_to_equity=_normalize_debt_to_equity(info.get("debtToEquity")),
        current_ratio=(
            float(info["currentRatio"])
            if isinstance(info.get("currentRatio"), (int, float))
            else None
        ),
        free_cash_flow_latest=fcf_latest,
        free_cash_flow_yoy_pct=_yoy(fcf, latest, prior),
        return_on_equity_pct=_normalize_pct(info.get("returnOnEquity")),
        payout_ratio_pct=payout,
        fcf_dividend_coverage=coverage,
    )


def merge_key_metrics_into_list(
    metrics: list[FundamentalMetric],
    key_rows: list[FundamentalMetric],
) -> list[FundamentalMetric]:
    """Replace key metric rows so the Financials page matches narrative values."""
    if not key_rows:
        return metrics

    key_labels = {row.label.lower() for row in key_rows}
    alias_labels = {
        "profit margin",
        "net margin",
        "debt/equity",
        "debt / equity",
    }
    filtered = [
        metric
        for metric in metrics
        if metric.label.lower() not in key_labels
        and metric.label.lower() not in alias_labels
    ]
    return key_rows + filtered


def apply_canonical_key_metrics(
    metrics: list[FundamentalMetric],
    canonical: CanonicalFinancialMetrics,
) -> list[FundamentalMetric]:
    return merge_key_metrics_into_list(metrics, canonical.to_key_metrics())


def _line_values(
    snapshot: FinancialStatementsSnapshot | None,
    label: str,
) -> dict[str, float | None]:
    if snapshot is None:
        return {}
    for section in (
        snapshot.income_statement,
        snapshot.balance_sheet,
        snapshot.cash_flow,
    ):
        for row in section:
            if row.label.lower() == label.lower():
                return dict(row.values)
    return {}


def _yoy(
    values: dict[str, float | None],
    latest: str | None,
    prior: str | None,
) -> float | None:
    if not latest or not prior:
        return None
    current = values.get(latest)
    previous = values.get(prior)
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / abs(previous)) * 100


def _margin_pct(
    numerator: dict[str, float | None],
    denominator: dict[str, float | None],
    latest: str | None,
) -> float | None:
    if not latest:
        return None
    rev = denominator.get(latest)
    num = numerator.get(latest)
    if rev is None or num is None or rev == 0:
        return None
    return (num / rev) * 100


def _normalize_pct(raw: float | None) -> float | None:
    if raw is None or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if abs(value) <= 1.5:
        return value * 100
    return value


def _normalize_debt_to_equity(raw: float | None) -> float | None:
    if raw is None or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    if value > 10:
        return value / 100.0
    return value


def _resolve_payout_ratio(info: dict[str, Any]) -> float | None:
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


def _fmt_compact_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    abs_val = abs(value)
    if abs_val >= 1_000_000_000_000:
        return f"{sign}${abs_val / 1_000_000_000_000:.1f}T"
    if abs_val >= 1_000_000_000:
        return f"{sign}${abs_val / 1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{sign}${abs_val / 1_000_000:.1f}M"
    return f"{sign}${abs_val:,.0f}"
