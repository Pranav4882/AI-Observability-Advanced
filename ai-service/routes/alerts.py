from fastapi import APIRouter, Query
from database.connection import db_pool
from database.queries import GET_RECENT_ALERTS, ACKNOWLEDGE_ALERT

router = APIRouter()

@router.get("/")
async def get_alerts(limit: int = Query(default=50, le=500)):
    rows = await db_pool.fetch(GET_RECENT_ALERTS, limit)
    return {"count": len(rows), "alerts": rows}

@router.post("/acknowledge/{alert_id}")
async def acknowledge_alert(alert_id: str, user: str = "operator"):
    await db_pool.execute(ACKNOWLEDGE_ALERT, alert_id, user)
    return {"acknowledged": True, "alert_id": alert_id}
