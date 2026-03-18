from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter
from typing import List
from app.models.schwab_models import SchwabAccounts


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
