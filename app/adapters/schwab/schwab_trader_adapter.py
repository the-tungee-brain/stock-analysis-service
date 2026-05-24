from typing import Dict, Optional, Literal, List, Any
import requests
from datetime import datetime, timedelta, timezone
from app.models.schwab_order_models import OrderStatus

MAX_ORDER_LOOKBACK_DAYS = 60


class SchwabTraderAdapter:
    def __init__(self, session: requests.Session, base_uri: str):
        self.base_uri = base_uri
        self.session = session
        self._account_hash_cache: dict[tuple[str, str], str] = {}

    def _get_auth_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def get_account_numbers(self, access_token: str) -> List[Dict[str, Any]]:
        url = f"{self.base_uri}/accounts/accountNumbers"
        response = self.session.get(
            url,
            headers=self._get_auth_headers(access_token),
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def resolve_account_hash(
        self,
        access_token: str,
        account_number: str,
    ) -> str:
        cache_key = (access_token, str(account_number))
        cached = self._account_hash_cache.get(cache_key)
        if cached:
            return cached

        for entry in self.get_account_numbers(access_token=access_token):
            if str(entry.get("accountNumber")) == str(account_number):
                hash_value = entry.get("hashValue")
                if not hash_value:
                    break
                self._account_hash_cache[cache_key] = hash_value
                return hash_value

        raise ValueError(
            f"No Schwab account hash found for account number {account_number!r}."
        )

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

    @staticmethod
    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _iso(dt: datetime) -> str:
        return SchwabTraderAdapter._to_utc(dt).strftime("%Y-%m-%dT%H:%M:%S.000Z")

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
        account_hash = self.resolve_account_hash(
            access_token=access_token,
            account_number=account_number,
        )
        url = f"{self.base_uri}/accounts/{account_hash}/orders"

        now = self._to_utc(datetime.now(timezone.utc))

        if from_time and to_time:
            start = self._to_utc(from_time)
            end = self._to_utc(to_time)
        elif days_back is not None:
            bounded_days = max(1, min(int(days_back), MAX_ORDER_LOOKBACK_DAYS))
            start = now - timedelta(days=bounded_days)
            end = now
        else:
            raise ValueError("Must provide either days_back or (from_time + to_time)")

        params: Dict[str, Any] = {
            "fromEnteredTime": self._iso(start),
            "toEnteredTime": self._iso(end),
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

        if not response.ok:
            detail = response.text.strip()
            message = f"Schwab orders request failed ({response.status_code})"
            if detail:
                message = f"{message}: {detail[:500]}"
            raise requests.HTTPError(message, response=response)

        return response.json()
