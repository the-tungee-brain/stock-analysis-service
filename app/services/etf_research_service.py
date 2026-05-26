from typing import Any

from app.adapters.securitiesdb.securitiesdb_adapter import SecuritiesDbAdapter
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.models.company_research_models import EtfHoldingItem, EtfHoldingsContext


class EtfResearchService:
    DEFAULT_HOLDINGS_LIMIT = 25

    def __init__(
        self,
        securitiesdb_adapter: SecuritiesDbAdapter,
        fundamentals_builder: FundamentalsBuilder,
    ):
        self.securitiesdb_adapter = securitiesdb_adapter
        self.fundamentals_builder = fundamentals_builder

    def build_holdings_context(
        self,
        symbol: str,
        *,
        holdings_limit: int | None = None,
    ) -> EtfHoldingsContext | None:
        payload = self.securitiesdb_adapter.get_etf_holdings(symbol=symbol)
        if payload is None:
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        meta = payload.get("meta")
        meta_dict = meta if isinstance(meta, dict) else {}

        resolved_limit = holdings_limit or self.DEFAULT_HOLDINGS_LIMIT
        resolved_limit = max(1, min(resolved_limit, 100))

        raw_holdings = data.get("holdings")
        holdings: list[EtfHoldingItem] = []
        if isinstance(raw_holdings, list):
            for item in raw_holdings[:resolved_limit]:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                weight = item.get("weight_pct")
                if not isinstance(name, str) or not isinstance(weight, (int, float)):
                    continue
                ticker = item.get("ticker")
                sector = item.get("sector")
                market_cap = self._format_market_cap(item.get("market_cap"))
                holdings.append(
                    EtfHoldingItem(
                        ticker=ticker.upper() if isinstance(ticker, str) else None,
                        name=name,
                        weight_pct=float(weight),
                        sector=sector if isinstance(sector, str) else None,
                        market_cap=market_cap,
                    )
                )

        sector_breakdown = self._parse_sector_breakdown(data.get("sector_breakdown"))
        fund_metrics = self.fundamentals_builder.build_etf_metrics(symbol=symbol)

        total_holdings = data.get("total_holdings")
        if not isinstance(total_holdings, int):
            total_holdings = len(holdings)

        return EtfHoldingsContext(
            ticker=str(data.get("ticker") or symbol).upper(),
            total_holdings=total_holdings,
            aum=self._format_aum(data.get("aum")),
            sector_breakdown=sector_breakdown,
            holdings=holdings,
            dividend_yield=fund_metrics.get("dividend_yield"),
            expense_ratio=fund_metrics.get("expense_ratio"),
            data_as_of=self._extract_data_as_of(meta_dict),
            confidence_score=self._extract_confidence_score(meta_dict),
        )

    @staticmethod
    def _parse_sector_breakdown(raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        breakdown: dict[str, float] = {}
        for sector, weight in raw.items():
            if isinstance(sector, str) and isinstance(weight, (int, float)):
                breakdown[sector] = float(weight)
        return breakdown

    @staticmethod
    def _format_aum(value: Any) -> str | None:
        if value is None or not isinstance(value, (int, float)):
            return None
        abs_val = abs(float(value))
        sign = "-" if value < 0 else ""
        if abs_val >= 1_000_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000_000:.1f}T"
        if abs_val >= 1_000_000_000:
            return f"{sign}${abs_val / 1_000_000_000:.1f}B"
        if abs_val >= 1_000_000:
            return f"{sign}${abs_val / 1_000_000:.1f}M"
        return f"{sign}${abs_val:,.0f}"

    @staticmethod
    def _format_market_cap(value: Any) -> str | None:
        return EtfResearchService._format_aum(value)

    @staticmethod
    def _extract_data_as_of(meta: dict[str, Any]) -> str | None:
        domains = meta.get("domains")
        if not isinstance(domains, dict):
            return None
        etf_holdings = domains.get("etf_holdings")
        if not isinstance(etf_holdings, dict):
            return None
        last_updated = etf_holdings.get("last_updated")
        return last_updated if isinstance(last_updated, str) else None

    @staticmethod
    def _extract_confidence_score(meta: dict[str, Any]) -> float | None:
        score = meta.get("confidence_score")
        if isinstance(score, (int, float)):
            return float(score)
        return None
