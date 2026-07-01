import time
import psutil
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from telemetry.custom_metrics import (
    PREDICTION_REQUESTS_TOTAL,
    PREDICTION_LATENCY,
    SYSTEM_CPU_USAGE,
    SYSTEM_MEMORY_USAGE,
    SYSTEM_GPU_USAGE,
    SYSTEM_GPU_MEMORY_USAGE,
    SYSTEM_GPU_TEMPERATURE,
    PREDICTION_CONFIDENCE,
    PREDICTIONS_PER_CLASS,
)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        path = request.url.path

        if path.startswith("/predict"):
            status = "success" if response.status_code < 400 else "error"
            model_name = "unknown"
            if "/text" in path:
                model_name = "sentiment"
            elif "/digits" in path:
                model_name = "mnist_digits"
            else:
                model_name = request.query_params.get("model", "fashion_mnist")

            PREDICTION_REQUESTS_TOTAL.labels(
                status=status,
                model_name=model_name,
                model_version="v2.0.0",
            ).inc()
            PREDICTION_LATENCY.labels(model_name=model_name).observe(duration)

        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        import random
        simulated_gpu = max(5.0, min(95.0, cpu * 0.7 + random.uniform(-2, 2)))
        simulated_gpu_mem = max(15.0, min(90.0, 30.0 + cpu * 0.2))
        simulated_gpu_temp = 40.0 + simulated_gpu * 0.45 + random.uniform(-1, 1)

        SYSTEM_CPU_USAGE.set(cpu)
        SYSTEM_MEMORY_USAGE.set(mem)
        SYSTEM_GPU_USAGE.set(simulated_gpu)
        SYSTEM_GPU_MEMORY_USAGE.set(simulated_gpu_mem)
        SYSTEM_GPU_TEMPERATURE.set(simulated_gpu_temp)
        return response
