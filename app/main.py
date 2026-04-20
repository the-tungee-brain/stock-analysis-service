from fastapi import FastAPI, Depends, APIRouter
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.api.get_account_positions_route import router as get_account_positions_router
from app.api.analyze_positions_by_symbol_route import (
    router as analyze_positions_by_symbol_router,
)
from app.api.health_check_route import router as health_check_router
from app.api.auth_schwab_callback_route import router as auth_schwab_callback_router
from app.api.auth_schwab_connect_route import router as auth_schwab_connect_router
from app.api.auth_schwab_status_route import router as auth_schwab_status_route
from app.dependencies.lifespan import lifespan
from app.auth.dependencies import get_current_user
from app.api.auth_google_callback_route import router as auth_google_callback_route
from app.api.get_company_news_route import router as get_company_news_route
from app.api.get_stock_data_route import router as get_stock_data_route
from app.api.get_company_snapshot_route import router as get_company_snapshot_route
from app.api.get_performance_snapshot_route import (
    router as get_performance_snapshot_route,
)
from app.api.get_stock_summary_route import router as get_stock_summary_route
from app.api.get_business_details_route import router as get_business_details_route

API_PREFIX = "/api/v1"
AUTH_SCHWAB_PREFIX = f"{API_PREFIX}/auth/schwab"
AUTH_GOOGLE_PREFIX = f"{API_PREFIX}/auth/google"

load_dotenv()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://powerpocket.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

app.include_router(health_check_router)
app.include_router(auth_google_callback_route, prefix=AUTH_GOOGLE_PREFIX)
app.include_router(auth_schwab_callback_router, prefix=AUTH_SCHWAB_PREFIX)

app.include_router(
    auth_schwab_connect_router,
    prefix=AUTH_SCHWAB_PREFIX,
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    auth_schwab_status_route,
    prefix=AUTH_SCHWAB_PREFIX,
    dependencies=[Depends(get_current_user)],
)

protected_api = APIRouter(
    prefix=API_PREFIX,
    dependencies=[Depends(get_current_user)],
)

protected_api.include_router(get_account_positions_router)
protected_api.include_router(analyze_positions_by_symbol_router)
protected_api.include_router(get_company_news_route)
protected_api.include_router(get_stock_data_route)
protected_api.include_router(get_company_snapshot_route)
protected_api.include_router(get_performance_snapshot_route)
protected_api.include_router(get_stock_summary_route)
protected_api.include_router(get_business_details_route)

app.include_router(protected_api)
