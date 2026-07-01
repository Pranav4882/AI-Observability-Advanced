import asyncio
import psutil
import time
from database.connection import db_pool
from database.redis_client import redis_manager
from database.queries import INSERT_SYSTEM_SNAPSHOT

class TelemetryService:
    def __init__(self):
        self.request_count = 0
        self.last_count_time = time.time()

    async def periodic_system_snapshot(self):
        while True:
            try:
                await asyncio.sleep(30)
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                
                import os
                import random
                disk_path = "C:\\" if os.name == "nt" else "/"
                disk = psutil.disk_usage(disk_path)
                
                simulated_gpu = max(5.0, min(95.0, cpu * 0.7 + random.uniform(-2, 2)))
                simulated_gpu_mem = max(15.0, min(90.0, 30.0 + cpu * 0.2))
                simulated_gpu_temp = 40.0 + simulated_gpu * 0.45 + random.uniform(-1, 1)
                
                now = time.time()
                elapsed = now - self.last_count_time
                rps = self.request_count / max(elapsed, 1.0)
                self.request_count = 0
                self.last_count_time = now

                await db_pool.execute(
                    INSERT_SYSTEM_SNAPSHOT, 
                    cpu, 
                    mem.percent, 
                    round(mem.used / 1024 / 1024, 1), 
                    disk.percent, 
                    simulated_gpu, 
                    simulated_gpu_mem, 
                    simulated_gpu_temp, 
                    0, 
                    round(rps, 2)
                )

                snapshot = {
                    "cpu_percent": cpu, 
                    "memory_percent": mem.percent,
                    "memory_used_mb": round(mem.used / 1024 / 1024, 1),
                    "disk_percent": disk.percent, 
                    "gpu_percent": round(simulated_gpu, 1),
                    "gpu_memory_percent": round(simulated_gpu_mem, 1),
                    "gpu_temperature_celsius": round(simulated_gpu_temp, 1),
                    "requests_per_sec": round(rps, 2)
                }
                await redis_manager.set_json("system_snapshot", snapshot, ttl=60)
                await redis_manager.publish("metrics_stream", {"type": "system_snapshot", **snapshot})
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"snapshot error: {e}")
                await asyncio.sleep(5)

    def record_request(self):
        self.request_count += 1

telemetry_service = TelemetryService()
