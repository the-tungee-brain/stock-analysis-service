import logging
from typing import List, Literal, Optional
from pydantic import ValidationError

from app.adapters.schwab.schwab_market_adapter import (
    ContractType,
    SchwabMarketAdapter,
    StrategyType,
)
from app.models.schwab_market_models import QuotesResponse
from app.models.schwab_option_chain_models import OptionChain

QuoteField = Literal["quote", "fundamental", "all"]
logger = logging.getLogger(__name__)


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

        if isinstance(raw_quote_data, dict) and "invalidSymbols" in raw_quote_data:
            invalid_symbols = _normalize_invalid_symbols(
                raw_quote_data.get("invalidSymbols")
            )
            for invalid_symbol in invalid_symbols:
                logger.warning(
                    "Provider symbol unavailable provider=%s endpoint=%s symbol=%s reason=%s",
                    "schwab",
                    "quotes",
                    invalid_symbol,
                    "invalidSymbols",
                )
            raw_quote_data = {
                key: value
                for key, value in raw_quote_data.items()
                if key != "invalidSymbols"
            }

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

        try:
            return OptionChain.model_validate(raw_option_chains)
        except ValidationError as exc:
            logger.error(
                "Option chain validation failed for %s: %s",
                symbol,
                exc,
            )
            raise


def _normalize_invalid_symbols(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip().upper()] if value.strip() else []
    if isinstance(value, list):
        symbols: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                symbols.append(item.strip().upper())
        return symbols
    return []
