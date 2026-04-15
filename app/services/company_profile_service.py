from app.builders.finnhub_builder import FinnhubBuilder
from app.models.company_research_models import ResearchSnapshot


class CompanyProfileService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_snapshot(self, symbol: str):
        profile = self.finnhub_builder.get_company_profile(symbol=symbol)
        quote = self.finnhub_builder.get_quote(symbol=symbol)
        market_cap = self._format_market_cap(profile.marketCapitalization)

        return ResearchSnapshot(
            symbol=profile.ticker or symbol.upper(),
            name=profile.name,
            sector=profile.finnhubIndustry,
            country=profile.country,
            price=quote.c,
            changePct=quote.dp if hasattr(quote, "dp") else 0.0,
            marketCap=market_cap,
        )

    def _format_market_cap(self, mc: float) -> str:
        if mc >= 1_000_000:
            return f"{mc / 1_000_000:.1f}T"
        if mc >= 1_000:
            return f"{mc / 1_000:.1f}B"
        return f"{mc:.0f}M"
