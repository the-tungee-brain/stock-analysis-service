from typing import Dict, Optional, Literal
import requests


class SchwabMarketAdapter:
    def __init__(self, session: requests.Session, base_uri: str):
        self.base_uri = base_uri
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
        fields: str,
        indicative: bool = False,
    ):
        url = f"{self.base_uri}/quotes"
        params = {"symbols": symbols, "indicative": indicative}
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
