from fastapi import APIRouter, Query
from typing import Optional
from services.drift_service import drift_service
from database.connection import db_pool
from database.queries import GET_RECENT_DRIFT

router = APIRouter()

@router.get("/status")
async def drift_status(model: Optional[str] = Query(None)):
    return await drift_service.get_status(model)

@router.get("/history")
async def drift_history(limit: int = Query(default=50, le=500)):
    rows = await db_pool.fetch(GET_RECENT_DRIFT, limit)
    return {"count": len(rows), "data": rows}

@router.post("/check")
async def manual_drift_check(model: Optional[str] = Query(None)):
    from models.model_loader import model_manager
    if model:
        result = await drift_service.run_drift_check(model, force=True)
        if result:
            drift_service.latest_results[model] = result
            return result
        return {"model": model, "message": "Not enough data or model not found."}

    results = {}
    for name in model_manager.models:
        r = await drift_service.run_drift_check(name, force=True)
        if r:
            drift_service.latest_results[name] = r
            results[name] = r
        else:
            results[name] = {"message": "Not enough data"}
    return {"models": results}
