from app.builders.finnhub_builder import FinnhubBuilder
from app.models.company_research_models import ResearchSnapshot
import yfinance as yf


class CompanyProfileService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_snapshot(self, symbol: str):
        profile = self.finnhub_builder.get_company_profile(symbol=symbol)
        quote = self.finnhub_builder.get_quote(symbol=symbol)
        market_cap = self._format_market_cap(mc=profile.marketCapitalization)
        change_pct = self._compute_change_pct(
            current=getattr(quote, "c", None),
            prev_close=getattr(quote, "pc", None),
        )
        low_52w, high_52w = self.get_52w_range_yf(symbol=symbol)
        range_52w = f"${low_52w:.0f} – ${high_52w:.0f}"

        return ResearchSnapshot(
            symbol=profile.ticker or symbol.upper(),
            name=profile.name,
            sector=profile.finnhubIndustry,
            country=profile.country,
            price=quote.c,
            changePct=change_pct,
            marketCap=market_cap,
            range52w=range_52w,
            logo=profile.logo,
            weburl=profile.weburl,
        )

    def _compute_change_pct(
        self, current: float | None, prev_close: float | None
    ) -> float:
        if current is None or prev_close in (None, 0):
            return 0.0
        return (current / prev_close - 1.0) * 100.0

    def get_52w_range_yf(self, symbol: str) -> tuple[float, float]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d")

        if hist.empty:
            raise ValueError(f"No historical data for {symbol}")

        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        return low_52w, high_52w

    def format_52w_range(self, symbol: str) -> str:
        low, high = self.get_52w_range_yf(symbol)
        return f"${low:.2f} – ${high:.2f}"

    def _format_market_cap(self, mc: float) -> str:
        if mc >= 1_000_000:
            return f"{mc / 1_000_000:.1f}T"
        if mc >= 1_000:
            return f"{mc / 1_000:.1f}B"
        return f"{mc:.0f}M"
