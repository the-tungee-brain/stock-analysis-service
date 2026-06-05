import logging
from typing import Dict, List, Optional

from app.adapters.schwab.schwab_market_adapter import (
    ContractType,
    SchwabUnsupportedSymbolError,
    StrategyType,
)
from app.broker.option_utils import option_chain_date_window
from app.builders.performance_builder import PerformanceBuilder
from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.models.company_research_models import PerformanceSnapshot
from app.models.schwab_market_models import PromptQuoteSnapshot
from app.models.schwab_option_chain_models import OptionChain

logger = logging.getLogger(__name__)


class MarketService:
    def __init__(
        self,
        schwab_market_builder: SchwabMarketBuilder,
        performance_builder: PerformanceBuilder,
    ):
        self.schwab_market_builder = schwab_market_builder
        self.performance_builder = performance_builder

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
    ) -> OptionChain | None:
        symbol_upper = symbol.strip().upper()
        if not from_date or not to_date:
            default_from, default_to = option_chain_date_window()
            from_date = from_date or default_from
            to_date = to_date or default_to

        try:
            return self.schwab_market_builder.get_option_chains(
                access_token=access_token,
                symbol=symbol_upper,
                contract_type=contract_type,
                strike_count=strike_count,
                include_underlying_quote=include_underlying_quote,
                from_date=from_date,
                to_date=to_date,
                strategy=strategy,
            )
        except SchwabUnsupportedSymbolError as exc:
            logger.warning(
                "Provider symbol unavailable provider=%s endpoint=%s symbol=%s reason=%s",
                "schwab",
                exc.endpoint,
                exc.symbol.strip().upper(),
                exc.reason,
            )
            return None

    def get_performance(self, symbol: str) -> PerformanceSnapshot:
        return self.performance_builder.build(symbol=symbol)
