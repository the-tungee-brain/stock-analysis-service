from fastapi import FastAPI, Depends, APIRouter
from dotenv import load_dotenv

from app.api.get_account_positions_route import router as get_account_positions_router
from app.api.analyze_positions_by_symbol_route import (
    router as analyze_positions_by_symbol_router,
)
from app.api.health_check_route import router as health_check_router
from app.api.auth_schwab_callback_route import router as auth_schwab_callback_router
from app.api.auth_schwab_connect_route import router as auth_schwab_connect_router
from app.dependencies.lifespan import lifespan
from app.auth.dependencies import get_current_user
from app.api.auth_google_callback_route import router as auth_google_callback_route

API_PREFIX = "/api/v1"
AUTH_SCHWAB_PREFIX = f"{API_PREFIX}/auth/schwab"
AUTH_GOOGLE_PREFIX = f"{API_PREFIX}/auth/google"

load_dotenv()

app = FastAPI(lifespan=lifespan)

app.include_router(health_check_router)
app.include_router(auth_google_callback_route, prefix=AUTH_GOOGLE_PREFIX)
app.include_router(auth_schwab_connect_router, prefix=AUTH_SCHWAB_PREFIX)
app.include_router(auth_schwab_callback_router, prefix=AUTH_SCHWAB_PREFIX)

protected_api = APIRouter(
    prefix=API_PREFIX,
    dependencies=[Depends(get_current_user)],
)

protected_api.include_router(get_account_positions_router)
protected_api.include_router(analyze_positions_by_symbol_router)

app.include_router(protected_api)
