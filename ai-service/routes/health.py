import psutil
from fastapi import APIRouter, Body
from models.model_loader import model_manager
from database.connection import db_pool
from database.redis_client import redis_manager

router = APIRouter()

@router.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "ai-observability",
        "models_loaded": list(model_manager.models.keys())
    }

@router.get("/readiness")
async def readiness():
    checks = {
        "models": {
            name: {
                "loaded": m.runtime_type != "none",
                "runtime": m.runtime_type,
                "version": m.model_version,
                "classes": m.num_classes,
                "has_baseline": m.training_baseline is not None
            }
            for name, m in model_manager.models.items()
        },
        "database": False,
        "redis": False,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 1)
        }
    }

    try:
        if db_pool.pool:
            checks["database"] = (await db_pool.fetchval("SELECT 1") == 1)
    except Exception: pass

    try:
        if redis_manager.client:
            await redis_manager.client.ping()
            checks["redis"] = True
    except Exception: pass

    all_ready = len(model_manager.models) > 0 and checks["database"] and checks["redis"]
    return {"ready": all_ready, "checks": checks}

@router.get("/db-redis-stats")
async def db_redis_stats():
    db_ok = False
    counts = {"inference_logs": 0, "drift_scores": 0, "alert_history": 0}
    recent_logs = []
    try:
        if db_pool.pool:
            counts["inference_logs"] = await db_pool.fetchval("SELECT COUNT(*) FROM inference_logs") or 0
            counts["drift_scores"] = await db_pool.fetchval("SELECT COUNT(*) FROM drift_scores") or 0
            counts["alert_history"] = await db_pool.fetchval("SELECT COUNT(*) FROM alert_history") or 0
            
            rows = await db_pool.fetch(
                "SELECT id, model_name, prediction, confidence, timestamp FROM inference_logs ORDER BY timestamp DESC LIMIT 5"
            )
            recent_logs = [
                {
                    "id": str(row["id"])[:8],
                    "model_name": row["model_name"],
                    "prediction": row["prediction"],
                    "confidence": float(row["confidence"]),
                    "timestamp": row["timestamp"].strftime("%H:%M:%S")
                }
                for row in rows
            ]
            db_ok = True
    except Exception as e:
        print(f"Stats DB fetch failed: {e}")

    redis_ok = False
    redis_keys = {}
    try:
        if redis_manager.client:
            redis_ok = True
            keys = await redis_manager.client.keys("requests:*")
            for key in keys:
                k_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                val = await redis_manager.client.get(key)
                val_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
                redis_keys[k_str] = int(val_str) if val_str.isdigit() else val_str
    except Exception as e:
        print(f"Stats Redis fetch failed: {e}")

    return {
        "postgres": {
            "status": "connected" if db_ok else "disconnected",
            "counts": counts,
            "recent_logs": recent_logs
        },
        "redis": {
            "status": "connected" if redis_ok else "disconnected",
            "keys_count": len(redis_keys),
            "keys_values": redis_keys
        }
    }
