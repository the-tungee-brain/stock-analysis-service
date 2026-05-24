from typing import Dict, List

from app.broker.option_utils import (
    summarize_assignment_risk_structural,
    summarize_csp_cash_reserves,
)
from app.broker.strategy_detector import detect_option_strategy
from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.models.schwab_models import Position, SchwabAccounts


class PortfolioService:
    def __init__(self, schwab_trader_builder: SchwabTraderBuilder):
        self.schwab_trader_builder = schwab_trader_builder

    def get_enriched_account(self, access_token: str) -> Dict[str, object]:
        account: SchwabAccounts = self.schwab_trader_builder.get_account(
            access_token=access_token
        )

        positions = self._annotate_option_strategies(
            account.securitiesAccount.positions
        )
        account = account.model_copy(
            update={
                "securitiesAccount": account.securitiesAccount.model_copy(
                    update={"positions": positions}
                )
            }
        )

        positions_by_symbol: Dict[str, List[Position]] = {
            symbol: [p for p in positions if self._symbol_key(p) == symbol]
            for symbol in {self._symbol_key(p) for p in positions}
        }

        cash_balance = account.securitiesAccount.currentBalances.cashBalance

        return {
            "account": account,
            "positions": positions_by_symbol,
            "cashSecuredPutSummary": summarize_csp_cash_reserves(
                positions=positions,
                cash_balance=cash_balance,
            ),
            "assignmentRiskSummary": summarize_assignment_risk_structural(
                positions=positions,
            ),
        }

    @staticmethod
    def _annotate_option_strategies(positions: List[Position]) -> List[Position]:
        annotated: List[Position] = []
        for position in positions:
            strategy = detect_option_strategy(position, positions)
            if strategy is None:
                annotated.append(position)
            else:
                annotated.append(position.model_copy(update={"optionStrategy": strategy}))
        return annotated

    def _symbol_key(self, pos: Position) -> str:
        if pos.instrument.assetType == "OPTION" and pos.instrument.underlyingSymbol:
            return pos.instrument.underlyingSymbol
        return pos.instrument.symbol
