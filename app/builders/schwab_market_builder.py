from app.adapters.schwab.schwab_market_adapter import SchwabMarketAdapter
from typing import List, Literal
from app.models.schwab_market_models import QuotesResponse

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
