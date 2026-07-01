from database.connection import db_pool
from database.redis_client import redis_manager
from database.queries import GET_RECENT_PREDICTIONS, GET_ERROR_RATE
import psutil

class MetricsCollector:
    async def get_summary(self) -> dict:
        cached = await redis_manager.get_json("metrics_summary")
        if cached: return cached

        summary = {
            "total_requests": 0, "avg_latency_ms": 0, "avg_confidence": 0, "error_rate": 0, "predictions_last_5min": 0,
            "system": {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent}
        }

        try:
            rows = await db_pool.fetch(GET_RECENT_PREDICTIONS)
            if rows:
                summary["predictions_last_5min"] = len(rows)
                summary["avg_latency_ms"] = round(sum(r["latency_ms"] for r in rows) / len(rows), 2)
                summary["avg_confidence"] = round(sum(r["confidence"] for r in rows) / len(rows), 4)

            error_data = await db_pool.fetchrow(GET_ERROR_RATE)
            if error_data and error_data.get("error_rate"):
                summary["error_rate"] = round(error_data["error_rate"], 4)
                summary["total_requests"] = error_data.get("total", 0)
        except Exception as e:
            print(f"metrics fetch failed: {e}")

        await redis_manager.set_json("metrics_summary", summary, ttl=10)
        return summary

metrics_collector = MetricsCollector()
