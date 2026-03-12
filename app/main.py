from fastapi import FastAPI
from app.api.get_account_positions_route import router as get_account_positions_route
from app.api.analyze_positions_by_symbol_route import (
    router as analyze_positions_by_symbol_route,
)
from app.api.health_check_route import router as health_check_route
from dotenv import load_dotenv
from app.dependencies.lifespan import lifespan

API_PREFIX = "/api/v1"

load_dotenv()

app = FastAPI(lifespan=lifespan)

app.include_router(get_account_positions_route, prefix=API_PREFIX)
app.include_router(analyze_positions_by_symbol_route, prefix=API_PREFIX)
app.include_router(health_check_route)
