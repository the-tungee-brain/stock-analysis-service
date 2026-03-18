from fastapi import APIRouter
from fastapi import Depends
from app.dependencies.service_dependencies import get_portfolio_service
from app.dependencies.service_dependencies import get_schwab_auth_service
from app.auth.dependencies import get_current_user_id
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService

router = APIRouter()


@router.get("/get-account-positions")
def get_account_positions(
    user_id: str = Depends(get_current_user_id),
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    schwab_token = schwab_auth_service.get_valid_token_by_user_id(user_id=user_id)

    account_map = portfolio_service.get_enriched_account(
        access_token=schwab_token.access_token
    )
    return {
        "schwab_positions": account_map["positions"],
        "account": account_map["account"],
    }
