from prometheus_client import Counter, Histogram, Gauge, Info

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total requests",
    ["status", "model_name", "model_version"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency",
    ["model_name"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Confidence",
    ["model_name"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

MODEL_INFO = Info("model", "Metadata")

PREDICTIONS_PER_CLASS = Counter(
    "predictions_per_class_total",
    "Predictions per class",
    ["model_name", "class_name", "class_id"]
)

DRIFT_PSI_SCORE = Gauge("drift_psi_score", "PSI", ["model_name"])
DRIFT_KL_DIVERGENCE = Gauge("drift_kl_divergence", "KL", ["model_name"])
DRIFT_IS_DETECTED = Gauge("drift_detected", "Drift status", ["model_name"])

SYSTEM_CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU")
SYSTEM_MEMORY_USAGE = Gauge("system_memory_usage_percent", "Memory")
SYSTEM_GPU_USAGE = Gauge("system_gpu_usage_percent", "GPU")
SYSTEM_GPU_MEMORY_USAGE = Gauge("system_gpu_memory_usage_percent", "GPU memory")
SYSTEM_GPU_TEMPERATURE = Gauge("system_gpu_temperature_celsius", "GPU temp")

PREDICTION_CLASS_IMBALANCE = Gauge(
    "prediction_class_imbalance",
    "Class imbalance",
    ["model_name"]
)

PREDICTION_ANOMALY_SCORE = Histogram(
    "prediction_anomaly_score",
    "Anomaly score",
    ["model_name"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

ALERTS_TOTAL = Counter("alerts_total", "Total alerts", ["alert_type", "severity"])
ACTIVE_WEBSOCKET_CONNECTIONS = Gauge("active_websocket_connections", "WebSocket connections")
