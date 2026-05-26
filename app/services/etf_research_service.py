from typing import Any

from app.adapters.securitiesdb.securitiesdb_adapter import SecuritiesDbAdapter
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.models.company_research_models import EtfHoldingItem, EtfHoldingsContext
from app.utils.etf_holdings_quality import (
    compute_quality_score,
    rank_etf_holdings_by_quality,
)


class EtfResearchService:
    DEFAULT_HOLDINGS_LIMIT = 25
    DEFAULT_QUALITY_LIMIT = 5

    def __init__(
        self,
        securitiesdb_adapter: SecuritiesDbAdapter,
        fundamentals_builder: FundamentalsBuilder,
    ):
        self.securitiesdb_adapter = securitiesdb_adapter
        self.fundamentals_builder = fundamentals_builder

    def is_etf_symbol(self, symbol: str) -> bool:
        payload = self.securitiesdb_adapter.get_etf_holdings(symbol=symbol)
        return payload is not None

    def build_holdings_context(
        self,
        symbol: str,
        *,
        holdings_limit: int | None = None,
        quality_limit: int | None = None,
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
        resolved_quality_limit = quality_limit or self.DEFAULT_QUALITY_LIMIT
        resolved_quality_limit = max(1, min(resolved_quality_limit, 10))

        raw_holdings = data.get("holdings")
        all_holdings: list[EtfHoldingItem] = []
        if isinstance(raw_holdings, list):
            for item in raw_holdings:
                if not isinstance(item, dict):
                    continue
                parsed = self._parse_holding_item(item)
                if parsed is not None:
                    all_holdings.append(parsed)

        strongest, weakest = rank_etf_holdings_by_quality(
            all_holdings,
            limit=resolved_quality_limit,
        )

        sector_breakdown = self._parse_sector_breakdown(data.get("sector_breakdown"))
        fund_metrics = self.fundamentals_builder.build_etf_metrics(symbol=symbol)

        total_holdings = data.get("total_holdings")
        if not isinstance(total_holdings, int):
            total_holdings = len(all_holdings)

        return EtfHoldingsContext(
            ticker=str(data.get("ticker") or symbol).upper(),
            total_holdings=total_holdings,
            aum=self._format_aum(data.get("aum")),
            sector_breakdown=sector_breakdown,
            holdings=all_holdings[:resolved_limit],
            strongest_holdings=strongest,
            weakest_holdings=weakest,
            dividend_yield=fund_metrics.get("dividend_yield"),
            expense_ratio=fund_metrics.get("expense_ratio"),
            data_as_of=self._extract_data_as_of(meta_dict),
            confidence_score=self._extract_confidence_score(meta_dict),
        )

    @staticmethod
    def _parse_holding_item(item: dict[str, Any]) -> EtfHoldingItem | None:
        name = item.get("name")
        weight = item.get("weight_pct")
        if not isinstance(name, str) or not isinstance(weight, (int, float)):
            return None

        ticker = item.get("ticker")
        sector = item.get("sector")
        piotroski_raw = item.get("piotroski_f")
        altman_raw = item.get("altman_z")
        piotroski_f = int(piotroski_raw) if isinstance(piotroski_raw, int) else None
        altman_z = float(altman_raw) if isinstance(altman_raw, (int, float)) else None

        return EtfHoldingItem(
            ticker=ticker.upper() if isinstance(ticker, str) else None,
            name=name,
            weight_pct=float(weight),
            sector=sector if isinstance(sector, str) else None,
            market_cap=EtfResearchService._format_market_cap(item.get("market_cap")),
            piotroski_f=piotroski_f,
            altman_z=altman_z,
            quality_score=compute_quality_score(piotroski_f, altman_z),
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
