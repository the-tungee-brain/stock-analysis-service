from typing import Dict, Optional, Literal
import requests


ContractType = Literal["ALL", "CALL", "PUT"]
StrategyType = Literal["SINGLE", "ANALYTICAL"]


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

        response = self.session.get(
            url,
            headers=self._get_auth_headers(access_token=access_token),
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
