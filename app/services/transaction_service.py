from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.models.schwab_order_models import SchwabOrder
from typing import List


class TransactionService:
    def __init__(self, schwab_trader_builder: SchwabTraderBuilder):
        self.schwab_trader_builder = schwab_trader_builder

    def get_filled_orders_by_symbol(
        self, account_number: str, access_token: str, symbol: str
    ) -> List[SchwabOrder]:
        schwab_orders = self.schwab_trader_builder.get_orders(
            account_number=account_number,
            access_token=access_token,
            status="FILLED",
            days_back=30,
        )

        symbol = symbol.upper()

        return [
            order
            for order in schwab_orders
            if any(
                leg.instrument
                and leg.instrument.symbol
                and leg.instrument.symbol.upper() == symbol
                for leg in (order.orderLegCollection or [])
            )
        ]
