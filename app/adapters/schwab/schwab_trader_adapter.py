from typing import Dict, Optional, Literal
import requests


class SchwabTraderAdapter:
    def __init__(self, session: requests.Session, base_uri: str):
        self.base_uri = base_uri
        self.session = session

    def _get_auth_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def get_accounts(
        self, access_token: str, fields: Optional[Literal["positions"]] = "positions"
    ):
        url = f"{self.base_uri}/accounts"
        params = {}
        if fields:
            params["fields"] = fields

        response = self.session.get(
            url,
            headers=self._get_auth_headers(access_token),
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()
