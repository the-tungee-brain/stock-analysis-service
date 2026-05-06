from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter
from typing import List, Optional
from app.models.schwab_models import SchwabAccounts
from app.models.schwab_order_models import OrderStatus, SchwabOrder
from datetime import datetime


class SchwabTraderBuilder:
    def __init__(self, schwab_trader_adapter: SchwabTraderAdapter):
        self.schwab_trader_adapter = schwab_trader_adapter

    def get_account(self, access_token: str) -> SchwabAccounts:
        data = self.schwab_trader_adapter.get_accounts(
            access_token=access_token, fields="positions"
        )
        schwab_accounts: List[SchwabAccounts] = [
            SchwabAccounts.model_validate(item) for item in data
        ]
        return schwab_accounts[0]

    def get_orders(
        self,
        account_number: str,
        access_token: str,
        status: Optional[OrderStatus] = None,
        days_back: Optional[int] = None,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None,
        max_results: int = 3000,
    ) -> List[SchwabOrder]:
        data = self.schwab_trader_adapter.get_orders(
            account_number=account_number,
            access_token=access_token,
            status=status,
            days_back=days_back,
            from_time=from_time,
            to_time=to_time,
            max_results=max_results,
        )
        return [SchwabOrder.model_validate(item) for item in data]
