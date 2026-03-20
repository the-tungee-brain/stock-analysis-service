from app.adapters.schwab.schwab_market_adapter import SchwabMarketAdapter
from typing import List, Literal, Optional
from app.models.schwab_market_models import QuotesResponse
from app.adapters.schwab.schwab_market_adapter import ContractType, StrategyType
from app.models.schwab_option_chain_models import OptionChain

QuoteField = Literal["quote", "fundamental", "all"]


class SchwabMarketBuilder:
    def __init__(self, schwab_market_adapter: SchwabMarketAdapter):
        self.schwab_market_adapter = schwab_market_adapter

    def get_quotes(
        self,
        access_token: str,
        symbols: List[str],
        fields: List[QuoteField] = [],
        indicative: bool = False,
    ) -> QuotesResponse:
        raw_quote_data = self.schwab_market_adapter.get_quotes(
            access_token=access_token,
            symbols=",".join(symbols),
            fields=",".join(fields),
            indicative=indicative,
        )

        return QuotesResponse.model_validate(raw_quote_data)

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
        raw_option_chains = self.schwab_market_adapter.get_option_chains(
            access_token=access_token,
            symbol=symbol,
            contract_type=contract_type,
            strike_count=strike_count,
            include_underlying_quote=include_underlying_quote,
            from_date=from_date,
            to_date=to_date,
            strategy=strategy,
        )

        return OptionChain.model_validate(raw_option_chains)
