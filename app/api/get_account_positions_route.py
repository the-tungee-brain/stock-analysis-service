from fastapi import APIRouter
from fastapi import Depends
from app.dependencies.service_dependencies import get_portfolio_service
from app.dependencies.service_dependencies import get_schwab_auth_service
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService

router = APIRouter()


@router.get("/get-account-positions")
def get_account_positions(
    portfolio_service: PortfolioService = Depends(get_portfolio_service),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    schwab_token = schwab_auth_service.get_access_token()

    schwab_positions = portfolio_service.get_account_positions(
        access_token=schwab_token.access_token
    )
    return {"schwab_positions": schwab_positions}
