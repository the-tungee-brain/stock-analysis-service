from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.models.schwab_market_models import InstrumentQuote, PromptQuoteSnapshot
from typing import Dict


class MarketService:
    def __init__(self, schwab_market_builder: SchwabMarketBuilder):
        self.schwab_market_builder = schwab_market_builder

    def get_enriched_quote_snapshot(
        self, access_token: str, symbol: str
    ) -> Dict[str, PromptQuoteSnapshot]:
        quotes_response = self.schwab_market_builder.get_quotes(
            access_token=access_token, symbols=[symbol]
        )
        try:
            instrument: InstrumentQuote = quotes_response.__root__[symbol]
        except KeyError:
            raise ValueError(f"No quote data returned for symbol {symbol!r}")

        q = instrument.quote
        ref = instrument.reference
        f = instrument.fundamental

        return {
            symbol: PromptQuoteSnapshot(
                symbol=instrument.symbol,
                asset_main_type=instrument.assetMainType,
                asset_sub_type=instrument.assetSubType,
                description=ref.description,
                last=q.lastPrice,
                net_change=q.netChange,
                net_change_pct=q.netPercentChange,
                high_52w=q._52WeekHigh,
                low_52w=q._52WeekLow,
                volume=q.totalVolume,
                avg_10d_volume=f.avg10DaysVolume if f else None,
                avg_1y_volume=f.avg1YearVolume if f else None,
                implied_vol=q.volatility,
            )
        }
