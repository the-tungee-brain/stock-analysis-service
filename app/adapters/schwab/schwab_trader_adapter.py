from typing import Dict, Optional, Literal, List, Any
import requests
from datetime import datetime, timedelta, timezone
from app.models.schwab_order_models import OrderStatus


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

    def get_orders(
        self,
        account_number: str,
        access_token: str,
        status: Optional[OrderStatus] = None,
        days_back: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        max_results: int = 3000,
    ) -> List[Dict[str, Any]]:

        url = f"{self.base_uri}/accounts/{account_number}/orders"

        now = datetime.now(timezone.utc)

        if from_time and to_time:
            start = from_time
            end = to_time
        elif days_back:
            start = now - timedelta(days=days_back)
            end = now
        else:
            raise ValueError("Must provide either days_back or (from_time + to_time)")

        def _iso(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        params: Dict[str, Any] = {
            "fromEnteredTime": _iso(start),
            "toEnteredTime": _iso(end),
            "maxResults": max_results,
        }

        if status:
            params["status"] = status

        response = self.session.get(
            url,
            headers=self._get_auth_headers(access_token),
            params=params,
            timeout=10,
        )

        response.raise_for_status()
        return response.json()
