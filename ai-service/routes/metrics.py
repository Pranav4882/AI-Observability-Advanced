from fastapi import APIRouter, Query
from database.connection import db_pool
from database.queries import GET_HOURLY_METRICS, GET_PER_MODEL_STATS
from services.metrics_service import metrics_collector

router = APIRouter()

@router.get("/")
async def get_metrics():
    summary = await metrics_collector.get_summary()
    try:
        summary["per_model"] = await db_pool.fetch(GET_PER_MODEL_STATS)
    except Exception:
        summary["per_model"] = []
    return summary

@router.get("/prometheus")
async def prometheus_metrics():
    from fastapi.responses import PlainTextResponse
    from prometheus_client import generate_latest
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type="text/plain; version=0.0.4")

@router.get("/history")
async def metrics_history(hours: int = Query(default=24, le=168)):
    rows = await db_pool.fetch(GET_HOURLY_METRICS, hours)
    return {"hours_requested": hours, "data": rows}
