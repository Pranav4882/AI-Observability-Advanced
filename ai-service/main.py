import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import settings
from database.connection import db_pool
from database.redis_client import redis_manager
from models.model_loader import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("starting up")
    model_manager.load_all(settings.models_dir)
    await db_pool.connect()
    await redis_manager.connect()

    from services.telemetry_service import telemetry_service
    from services.drift_service import drift_service
    telemetry_task = asyncio.create_task(telemetry_service.periodic_system_snapshot())
    drift_task = asyncio.create_task(drift_service.periodic_drift_check())
    print(f"loaded models: {list(model_manager.models.keys())}")
    yield
    print("Shutting down...")
    telemetry_task.cancel()
    drift_task.cancel()
    await db_pool.disconnect()
    await redis_manager.disconnect()

app = FastAPI(
    title="AI Observability Platform",
    description="ML model monitoring and observability API.",
    version="2.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from telemetry.middleware import MetricsMiddleware
app.add_middleware(MetricsMiddleware)

from routes.predict import router as predict_router
from routes.metrics import router as metrics_router
from routes.health import router as health_router
from routes.drift import router as drift_router
from routes.alerts import router as alerts_router
from routes.websocket import router as ws_router

app.include_router(predict_router, prefix="/predict", tags=["Prediction"])
app.include_router(metrics_router, prefix="/metrics", tags=["Metrics"])
app.include_router(health_router, tags=["Health"])
app.include_router(drift_router, prefix="/drift", tags=["Drift Detection"])
app.include_router(alerts_router, prefix="/alerts", tags=["Alerts"])
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])

from fastapi.staticfiles import StaticFiles
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

@app.get("/")
async def root():
    return {
        "service": "AI Observability Platform",
        "version": "2.0.0",
        "status": "running",
        "models_loaded": list(model_manager.models.keys()),
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
