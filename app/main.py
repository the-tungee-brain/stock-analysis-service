from fastapi import FastAPI, Depends, APIRouter
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.api.get_recent_orders_route import router as get_recent_orders_router
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
from app.api.get_fundamentals_route import router as get_fundamentals_route
from app.api.sec_research_routes import router as sec_research_router
from app.api.get_earnings_route import router as get_earnings_route
from app.api.search_symbols_route import router as search_symbols_route
from app.api.research_chat_route import router as research_chat_router
from app.api.chat_sessions_route import router as chat_sessions_router
from app.api.get_account_positions_route import router as get_account_positions_router
from app.api.get_portfolio_brief_route import router as get_portfolio_brief_router
from app.api.get_symbol_analysis_precomputed_route import (
    router as get_symbol_analysis_precomputed_router,
)
from app.api.get_symbol_intelligence_route import (
    router as get_symbol_intelligence_router,
)
from app.api.get_option_chain_debug_route import router as get_option_chain_debug_router
from app.api.internal_morning_brief_route import router as internal_morning_brief_router
from app.api.portfolio_memory_routes import router as portfolio_memory_router
from app.api.strategy_routes import router as strategy_router

API_PREFIX = "/api/v1"
AUTH_SCHWAB_PREFIX = f"{API_PREFIX}/auth/schwab"
AUTH_GOOGLE_PREFIX = f"{API_PREFIX}/auth/google"

load_dotenv()

app = FastAPI(lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "https://tomcrest.com",
    "https://www.tomcrest.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

app.include_router(health_check_router)
app.include_router(internal_morning_brief_router, prefix=API_PREFIX)
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
protected_api.include_router(get_portfolio_brief_router)
protected_api.include_router(portfolio_memory_router)
protected_api.include_router(strategy_router)
protected_api.include_router(get_symbol_intelligence_router)
protected_api.include_router(get_symbol_analysis_precomputed_router)
protected_api.include_router(get_option_chain_debug_router)
protected_api.include_router(get_recent_orders_router)
protected_api.include_router(analyze_positions_by_symbol_router)
protected_api.include_router(get_company_news_route)
protected_api.include_router(get_stock_data_route)
protected_api.include_router(get_company_snapshot_route)
protected_api.include_router(get_performance_snapshot_route)
protected_api.include_router(get_stock_summary_route)
protected_api.include_router(get_business_details_route)
protected_api.include_router(get_fundamentals_route)
protected_api.include_router(sec_research_router)
protected_api.include_router(get_earnings_route)
protected_api.include_router(search_symbols_route)
protected_api.include_router(research_chat_router)
protected_api.include_router(chat_sessions_router)

app.include_router(protected_api)
