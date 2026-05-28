from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.yahoo_snapshot_time import yahoo_snapshot_as_of
from app.models.yfinance_analysis_models import (
    AnalystPriceTargets,
    AnalystRatingAction,
    InstitutionalHolder,
    InsiderTransactionRow,
    OwnershipSnapshot,
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

_ESTIMATE_PERIOD_KEYS = ("0q", "+1q", "0y", "+1y")
# Yahoo can return many rows; cap payload size while exposing full history in the UI.
_MAX_INSIDER_TRANSACTIONS = 500


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
        rating_trend_headline = self._rating_trend_headline(raw.get("recommendations"))
        eps_estimates = self._parse_all_period_estimates(raw.get("earnings_estimate"))
        revenue_estimates = self._parse_all_period_estimates(
            raw.get("revenue_estimate")
        )
        next_q_eps = self._estimate_for_period(eps_estimates, "+1q")
        next_q_revenue = self._estimate_for_period(revenue_estimates, "+1q")
        revision_headline = self._revision_headline(raw.get("eps_revisions"), "+1q")
        drift_headline = self._eps_trend_headline(raw.get("eps_trend"), "+1q")
        growth_headline = self._growth_context_headline(raw.get("growth_estimates"), "+1y")
        recent_actions = self._parse_recent_rating_actions(
            raw.get("upgrades_downgrades")
        )
        ownership = self._parse_ownership(raw)

        if not any(
            [
                price_targets,
                recommendation,
                next_q_eps,
                next_q_revenue,
                eps_estimates,
                revenue_estimates,
                revision_headline,
                drift_headline,
                growth_headline,
                recent_actions,
                ownership,
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
            eps_estimates=eps_estimates,
            revenue_estimates=revenue_estimates,
            estimate_revision_headline=revision_headline,
            estimate_drift_headline=drift_headline,
            growth_context_headline=growth_headline,
            rating_trend_headline=rating_trend_headline,
            recent_rating_actions=recent_actions,
            ownership=ownership,
            data_as_of=yahoo_snapshot_as_of(),
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

    def _rating_trend_headline(self, df: Any) -> str | None:
        if df is None or not isinstance(df, pd.DataFrame) or len(df) < 2:
            return None

        working = df.copy()
        if "period" in working.columns:
            working = working.sort_values("period", ascending=True)

        latest_row = working.iloc[-1]
        prior_row = working.iloc[-2]
        latest_bull, latest_total = self._recommendation_bullish_share(latest_row)
        prior_bull, prior_total = self._recommendation_bullish_share(prior_row)
        if latest_total <= 0 or prior_total <= 0:
            return None

        delta = latest_bull - prior_bull
        if abs(delta) < 3:
            return (
                f"Analyst buy-side share steady at {latest_bull:.0f}% "
                f"(was {prior_bull:.0f}%)."
            )
        if delta > 0:
            return (
                f"Analyst sentiment shifted more bullish: buy-side share "
                f"{latest_bull:.0f}% vs {prior_bull:.0f}% prior period."
            )
        return (
            f"Analyst sentiment shifted more cautious: buy-side share "
            f"{latest_bull:.0f}% vs {prior_bull:.0f}% prior period."
        )

    def _recommendation_bullish_share(self, row: Any) -> tuple[float, int]:
        def col(name: str) -> int:
            for key in (name, name.replace("_", "")):
                if key in row.index:
                    value = row[key]
                    if value is not None and not pd.isna(value):
                        return int(value)
            return 0

        strong_buy = col("strongBuy")
        buy = col("buy")
        hold = col("hold")
        sell = col("sell")
        strong_sell = col("strongSell")
        total = strong_buy + buy + hold + sell + strong_sell
        if total <= 0:
            return 0.0, 0
        bullish = ((strong_buy + buy) / total) * 100
        return bullish, total

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

    def _parse_all_period_estimates(self, table: Any) -> list[PeriodEstimate]:
        estimates: list[PeriodEstimate] = []
        for period_key in _ESTIMATE_PERIOD_KEYS:
            estimate = self._parse_period_estimate(table, period_key)
            if estimate is not None:
                estimates.append(estimate)
        return estimates

    @staticmethod
    def _estimate_for_period(
        estimates: list[PeriodEstimate], period_key: str
    ) -> PeriodEstimate | None:
        for estimate in estimates:
            if estimate.period_key == period_key:
                return estimate
        return None

    def _parse_period_estimate(
        self,
        table: Any,
        period_key: str,
        *,
        year_ago_key: str = "",
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

    def _eps_trend_headline(self, table: Any, period_key: str) -> str | None:
        row = self._table_row(table, period_key)
        if row is None:
            return None

        current = self._optional_float(row.get("current"))
        if current is None:
            return None

        ago30 = self._optional_float(row.get("30daysAgo"))
        label = _PERIOD_LABELS.get(period_key, period_key).lower()
        if ago30 is None or ago30 == 0:
            return f"{label.capitalize()} EPS consensus is ${current:.2f}."

        pct = ((current - ago30) / abs(ago30)) * 100
        if abs(pct) < 0.05:
            drift = "unchanged"
        elif pct > 0:
            drift = f"up {pct:.1f}%"
        else:
            drift = f"down {abs(pct):.1f}%"

        return (
            f"{label.capitalize()} EPS consensus is ${current:.2f} ({drift} vs "
            f"30 days ago at ${ago30:.2f})."
        )

    def _parse_recent_rating_actions(
        self, df: Any, *, limit: int = 8
    ) -> list[AnalystRatingAction]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        try:
            sorted_df = df.sort_index(ascending=False).head(limit)
        except Exception:
            sorted_df = df.head(limit)

        actions: list[AnalystRatingAction] = []
        for idx, row in sorted_df.iterrows():
            date_str = self._index_to_iso_date(idx)
            if date_str is None:
                continue

            firm = self._row_str(row, "Firm", "firm")
            to_grade = self._row_str(row, "To Grade", "toGrade", "to_grade")
            if not firm or not to_grade:
                continue

            actions.append(
                AnalystRatingAction(
                    date=date_str,
                    firm=firm,
                    to_grade=to_grade,
                    from_grade=self._row_str(row, "From Grade", "fromGrade", "from_grade"),
                    action=self._row_str(row, "Action", "action"),
                )
            )
        return actions

    @staticmethod
    def _cell_to_iso_date(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            parsed = pd.Timestamp(value)
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        if parsed.year < 1990:
            return None
        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _index_to_iso_date(idx: Any) -> str | None:
        if isinstance(idx, (int, float)) and not isinstance(idx, bool):
            return None
        return YFinanceAnalysisBuilder._cell_to_iso_date(idx)

    def _growth_context_headline(self, table: Any, period_key: str) -> str | None:
        row = self._table_row(table, period_key)
        if row is None:
            return None

        stock = self._normalize_growth_pct(row.get("stock"))
        industry = self._normalize_growth_pct(row.get("industry"))
        sector = self._normalize_growth_pct(row.get("sector"))
        if stock is None:
            return None

        label = _PERIOD_LABELS.get(period_key, period_key).lower()
        parts = [f"{label.capitalize()} growth estimate {stock:.1f}%"]
        if industry is not None:
            parts.append(f"vs industry {industry:.1f}%")
        elif sector is not None:
            parts.append(f"vs sector {sector:.1f}%")
        return " ".join(parts) + "."

    @staticmethod
    def _normalize_growth_pct(value: Any) -> float | None:
        parsed = YFinanceAnalysisBuilder._optional_float(value)
        if parsed is None:
            return None
        if abs(parsed) <= 1.5:
            return parsed * 100
        return parsed

    def _parse_ownership(self, raw: dict[str, Any]) -> OwnershipSnapshot | None:
        insiders_pct, institutions_pct = self._parse_major_holder_pcts(
            raw.get("major_holders")
        )
        top_institutional = self._parse_institutional_holders(
            raw.get("institutional_holders")
        )
        insider_rows = self._parse_insider_transactions(raw.get("insider_transactions"))

        if (
            insiders_pct is None
            and institutions_pct is None
            and not top_institutional
            and not insider_rows
        ):
            return None

        return OwnershipSnapshot(
            insiders_pct_held=insiders_pct,
            institutions_pct_held=institutions_pct,
            top_institutional=top_institutional,
            recent_insider_transactions=insider_rows,
        )

    def _parse_major_holder_pcts(
        self, df: Any
    ) -> tuple[float | None, float | None]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None, None

        insiders_pct: float | None = None
        institutions_pct: float | None = None
        value_col = df.columns[0]

        for idx, row in df.iterrows():
            label = str(idx).lower()
            if not label or label == "nan":
                label_col = df.columns[0]
                label = str(row[label_col]).lower() if label_col in row.index else ""
            value = self._parse_holder_pct_value(row[value_col])
            if value is None:
                continue
            if "insider" in label:
                insiders_pct = value
            elif "institution" in label:
                institutions_pct = value
        return insiders_pct, institutions_pct

    @staticmethod
    def _parse_holder_pct_value(value: Any) -> float | None:
        """Parse insider/institution % held from Yahoo major_holders."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        raw = str(value).strip()
        had_percent_sign = "%" in raw
        text = raw.replace("%", "")
        try:
            parsed = float(text)
        except ValueError:
            return None

        if had_percent_sign:
            return YFinanceAnalysisBuilder._scale_pct_above_100(parsed)

        if 0 < parsed <= 1.5:
            return parsed * 100
        return YFinanceAnalysisBuilder._scale_pct_above_100(parsed)

    @staticmethod
    def _scale_pct_above_100(value: float) -> float:
        scaled = value
        while scaled > 100:
            scaled /= 100
        return scaled

    @staticmethod
    def _parse_percent_string(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip().replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_institutional_holders(
        self, df: Any, *, limit: int = 5
    ) -> list[InstitutionalHolder]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        pct_col = self._find_column(df, "% Out", "pctHeld", "pct_held")
        holder_col = self._find_column(df, "Holder", "holder")
        shares_col = self._find_column(df, "Shares", "shares")
        value_col = self._find_column(df, "Value", "value")

        working = df.copy()
        if pct_col:
            working["_pct"] = pd.to_numeric(
                working[pct_col].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            )
            working = working.sort_values("_pct", ascending=False)

        holders: list[InstitutionalHolder] = []
        for _, row in working.head(limit).iterrows():
            name = (
                str(row[holder_col]).strip()
                if holder_col and row.get(holder_col) is not None
                else ""
            )
            if not name:
                continue
            holders.append(
                InstitutionalHolder(
                    holder=name,
                    pct_held=(
                        self._parse_percent_string(row[pct_col])
                        if pct_col
                        else None
                    ),
                    shares=(
                        self._optional_float(row[shares_col]) if shares_col else None
                    ),
                    value=(
                        self._optional_float(row[value_col]) if value_col else None
                    ),
                )
            )
        return holders

    def _parse_insider_transactions(self, df: Any) -> list[InsiderTransactionRow]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        date_col = self._find_column(
            df, "Start Date", "startDate", "Date", "Filing Date", "startdate"
        )
        working = df.copy()
        if date_col:
            working["_sort_date"] = pd.to_datetime(working[date_col], errors="coerce")
            working = working.sort_values("_sort_date", ascending=False)
        else:
            try:
                working = df.sort_index(ascending=False)
            except Exception:
                working = df

        insider_col = self._find_column(working, "Insider", "insider", "Insider Trading")
        transaction_col = self._find_column(
            working, "Transaction", "transaction", "Text"
        )
        shares_col = self._find_column(working, "Shares", "shares")
        value_col = self._find_column(working, "Value", "value", "Value ($)")

        rows: list[InsiderTransactionRow] = []
        for idx, row in working.head(_MAX_INSIDER_TRANSACTIONS).iterrows():
            date_str = (
                self._cell_to_iso_date(row[date_col])
                if date_col and date_col in row.index
                else self._index_to_iso_date(idx)
            )
            if date_str is None:
                continue
            insider = (
                str(row[insider_col]).strip()
                if insider_col and row.get(insider_col) is not None
                else ""
            )
            if not insider:
                continue
            rows.append(
                InsiderTransactionRow(
                    date=date_str,
                    insider=insider,
                    transaction=(
                        self._row_str(row, "Transaction", "transaction", "Text")
                        if transaction_col
                        else None
                    ),
                    shares=(
                        self._optional_float(row[shares_col]) if shares_col else None
                    ),
                    value=(
                        self._optional_float(row[value_col]) if value_col else None
                    ),
                )
            )
        return rows

    @staticmethod
    def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
        for name in candidates:
            for col in df.columns:
                if str(col).lower().replace(" ", "") == name.lower().replace(" ", ""):
                    return col
        return None

    @staticmethod
    def _row_str(row: Any, *candidates: str) -> str | None:
        for name in candidates:
            for key in row.index:
                if str(key).lower().replace(" ", "") == name.lower().replace(" ", ""):
                    value = row[key]
                    if value is None or (isinstance(value, float) and pd.isna(value)):
                        continue
                    text = str(value).strip()
                    if text:
                        return text
        return None
