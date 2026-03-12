from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.models.schwab_models import Position
from typing import List, Dict


class PortfolioService:
    def __init__(self, schwab_trader_builder: SchwabTraderBuilder):
        self.schwab_trader_builder = schwab_trader_builder

    def get_account_positions(self, access_token: str) -> Dict[str, List[Position]]:
        positions = self.schwab_trader_builder.get_account_positions(
            access_token=access_token
        )

        return {
            symbol: [p for p in positions if self._symbol_key(p) == symbol]
            for symbol in {self._symbol_key(p) for p in positions}
        }

    def _symbol_key(self, pos: Position) -> str:
        if pos.instrument.assetType == "OPTION" and pos.instrument.underlyingSymbol:
            return pos.instrument.underlyingSymbol
        return pos.instrument.symbol
