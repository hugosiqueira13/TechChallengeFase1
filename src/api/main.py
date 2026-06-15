"""FastAPI — API de inferência de churn (Telco Customer Churn)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status

from src.api.middleware import LatencyMiddleware
from src.api.predictor import Predictor
from src.api.schemas import CustomerFeatures, HealthResponse, PredictResponse
from src.utils.logging_config import configure_logging

configure_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Inicializando Predictor...")
    try:
        _predictor.load()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Modelo não carregado na inicialização: %s", exc)
    yield
    logger.info("API encerrada.")


app = FastAPI(
    title="Telco Churn API",
    description="Prediz probabilidade de churn de clientes usando MLP treinado com PyTorch.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
async def health() -> dict:
    """Verifica se a API e o modelo estão prontos."""
    return {
        "status": "ok",
        "model_loaded": _predictor.is_ready,
        "version": app.version,
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inferência"],
)
async def predict(payload: CustomerFeatures) -> dict:
    """Retorna probabilidade de churn e tier de risco para um cliente."""
    if not _predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível. Execute o treinamento primeiro.",
        )
    try:
        return _predictor.predict(payload.model_dump())
    except Exception as exc:
        logger.exception("Erro durante predição")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def serve() -> None:
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    serve()
