"""FastAPI service for daily trend predictions."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from models.artifact_store import resolve_artifact_dir
from models.prediction_service import (
    LoadedModel,
    health_payload,
    load_deployed_model,
    predict_for_symbol,
)


def create_app(artifact_dir: Path | str | None = None) -> FastAPI:
    resolved_dir = resolve_artifact_dir(artifact_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.loaded_model = load_deployed_model(resolved_dir)
        app.state.artifact_dir = resolved_dir
        yield

    app = FastAPI(title="Stock Pattern Recognition API", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded: LoadedModel = app.state.loaded_model
        return health_payload(loaded)

    @app.get("/predict")
    def predict(symbol: str = Query(..., min_length=1)) -> dict[str, Any]:
        loaded: LoadedModel = app.state.loaded_model
        try:
            return predict_for_symbol(symbol, loaded)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()


def main() -> None:
    host = os.environ.get("PATTERN_API_HOST", "0.0.0.0")
    port = int(os.environ.get("PATTERN_API_PORT", "8080"))
    uvicorn.run("api.main:app", host=host, port=int(port), reload=False)


if __name__ == "__main__":
    main()
