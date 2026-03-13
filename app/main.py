from fastapi import FastAPI, Depends
from dotenv import load_dotenv

from app.api.get_account_positions_route import router as get_account_positions_router
from app.api.analyze_positions_by_symbol_route import (
    router as analyze_positions_by_symbol_router,
)
from app.api.health_check_route import router as health_check_router
from app.api.auth_schwab_callback_route import router as auth_schwab_callback_router
from app.api.auth_schwab_connect_route import router as auth_schwab_connect_router
from app.dependencies.lifespan import lifespan
from app.auth.dependencies import get_current_user, allow_anonymous
from app.api.auth_google_callback_route import router as auth_google_callback_route

API_PREFIX = "/api/v1"
AUTH_SCHWAB_PREFIX = f"{API_PREFIX}/auth/schwab"
AUTH_GOOGLE_PREFIX = f"{API_PREFIX}/auth/google"

load_dotenv()

app = FastAPI(
    lifespan=lifespan,
    dependencies=[Depends(get_current_user)],
)

app.include_router(
    health_check_router,
    dependencies=[Depends(allow_anonymous)],
)
app.include_router(
    auth_google_callback_route,
    prefix=AUTH_GOOGLE_PREFIX,
    dependencies=[Depends(allow_anonymous)],
)
app.include_router(
    auth_schwab_connect_router,
    prefix=AUTH_SCHWAB_PREFIX,
    dependencies=[Depends(allow_anonymous)],
)
app.include_router(
    auth_schwab_callback_router,
    prefix=AUTH_SCHWAB_PREFIX,
    dependencies=[Depends(allow_anonymous)],
)

app.include_router(get_account_positions_router, prefix=API_PREFIX)
app.include_router(analyze_positions_by_symbol_router, prefix=API_PREFIX)
