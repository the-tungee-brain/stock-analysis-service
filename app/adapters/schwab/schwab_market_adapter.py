from typing import Dict, Optional, Literal
import requests

from app.core.latency_observability import observe_dependency


ContractType = Literal["ALL", "CALL", "PUT"]
StrategyType = Literal["SINGLE", "ANALYTICAL"]


class SchwabUnsupportedSymbolError(Exception):
    def __init__(
        self,
        *,
        endpoint: str,
        symbol: str,
        status_code: int,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.endpoint = endpoint
        self.symbol = symbol
        self.status_code = status_code
        self.reason = reason


class SchwabMarketAdapter:
    def __init__(self, session: requests.Session, base_uri: str):
        self.base_uri = base_uri.rstrip("/")
        self.session = session

    def _get_auth_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def get_quotes(
        self,
        access_token: str,
        symbols: str,
        fields: Optional[str] = None,
        indicative: bool = False,
    ):
        url = f"{self.base_uri}/quotes"
        params: Dict[str, object] = {"symbols": symbols, "indicative": indicative}
        if fields:
            params["fields"] = fields

        with observe_dependency("schwab"):
            response = self.session.get(
                url,
                headers=self._get_auth_headers(access_token=access_token),
                params=params,
                timeout=10,
            )
        response.raise_for_status()
        return response.json()

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
    ):
        url = f"{self.base_uri}/chains"

        params: Dict[str, object] = {
            "symbol": symbol,
            "contractType": contract_type,
            "strikeCount": strike_count,
            "includeUnderlyingQuote": include_underlying_quote,
            "strategy": strategy,
        }
        if from_date:
            params["fromDate"] = from_date
        if to_date:
            params["toDate"] = to_date

        with observe_dependency("schwab"):
            response = self.session.get(
                url,
                headers=self._get_auth_headers(access_token=access_token),
                params=params,
                timeout=10,
            )
        if response.status_code == 400:
            raise SchwabUnsupportedSymbolError(
                endpoint="option_chains",
                symbol=symbol,
                status_code=response.status_code,
                reason="bad_request_invalid_or_unsupported_symbol",
            )
        response.raise_for_status()
        return response.json()
