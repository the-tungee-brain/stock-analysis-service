from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.models.schwab_market_models import PromptQuoteSnapshot
from typing import Dict, List, Optional
from app.adapters.schwab.schwab_market_adapter import ContractType, StrategyType
from app.models.schwab_option_chain_models import OptionChain


class MarketService:
    def __init__(self, schwab_market_builder: SchwabMarketBuilder):
        self.schwab_market_builder = schwab_market_builder

    def get_enriched_quote_snapshot(
        self, access_token: str, symbols: List[str]
    ) -> Dict[str, PromptQuoteSnapshot]:
        quotes_response = self.schwab_market_builder.get_quotes(
            access_token=access_token, symbols=symbols
        )

        snapshots: Dict[str, PromptQuoteSnapshot] = {}

        for symbol, instrument in quotes_response.root.items():
            q = instrument.quote
            ref = instrument.reference
            f = instrument.fundamental

            snapshots[symbol] = PromptQuoteSnapshot(
                symbol=instrument.symbol,
                asset_main_type=instrument.assetMainType,
                asset_sub_type=instrument.assetSubType,
                description=ref.description,
                last=q.lastPrice,
                net_change=q.netChange,
                net_change_pct=q.netPercentChange,
                high_52w=q.week_high_52,
                low_52w=q.week_low_52,
                volume=q.totalVolume,
                avg_10d_volume=f.avg10DaysVolume if f else None,
                avg_1y_volume=f.avg1YearVolume if f else None,
                implied_vol=q.volatility,
            )

        return snapshots

    def get_option_chains(
        self,
        access_token: str,
        symbol: str,
        contract_type: ContractType = "ALL",
        strike_count: int = 10,
        include_underlying_quote: bool = True,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        strategy: StrategyType = "SINGLE",
    ) -> OptionChain:
        return self.schwab_market_builder.get_option_chains(
            access_token=access_token,
            symbol=symbol,
            contract_type=contract_type,
            strike_count=strike_count,
            include_underlying_quote=include_underlying_quote,
            from_date=from_date,
            to_date=to_date,
            strategy=strategy,
        )
