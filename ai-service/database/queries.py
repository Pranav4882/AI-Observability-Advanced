INSERT_INFERENCE_LOG = """
    INSERT INTO inference_logs (
        request_id, model_version, input_shape, prediction, predicted_class, confidence, 
        top_3_classes, latency_ms, preprocess_ms, inference_ms, postprocess_ms,
        cpu_percent, memory_percent, runtime_type, status, error_message, model_name
    ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
"""

INSERT_DRIFT_SCORE = """
    INSERT INTO drift_scores (
        model_version, psi_score, kl_divergence, chi_square_stat, chi_square_p, 
        is_drifted, drift_severity, sample_size, training_dist, production_dist
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb)
"""

INSERT_ALERT = """
    INSERT INTO alert_history (alert_type, severity, message, metric_value, threshold_value)
    VALUES ($1, $2, $3, $4, $5)
"""

INSERT_SYSTEM_SNAPSHOT = """
    INSERT INTO system_snapshots (cpu_percent, memory_percent, memory_used_mb, disk_percent, gpu_percent, gpu_memory_percent, gpu_temp, active_connections, requests_per_sec)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
"""

GET_RECENT_PREDICTIONS = """
    SELECT predicted_class, confidence, latency_ms, timestamp, model_name FROM inference_logs
    WHERE timestamp > NOW() - INTERVAL '5 minutes' ORDER BY timestamp DESC
"""

GET_RECENT_PREDICTIONS_BY_MODEL = """
    SELECT predicted_class, confidence, latency_ms, timestamp, model_name FROM inference_logs
    WHERE timestamp > NOW() - INTERVAL '5 minutes' AND model_name = $1 ORDER BY timestamp DESC
"""

GET_RECENT_DRIFT = "SELECT * FROM drift_scores ORDER BY timestamp DESC LIMIT $1"
GET_RECENT_ALERTS = "SELECT * FROM alert_history ORDER BY timestamp DESC LIMIT $1"
GET_HOURLY_METRICS = "SELECT * FROM hourly_metrics LIMIT $1"

GET_ERROR_RATE = """
    SELECT COUNT(*) FILTER (WHERE status = 'error')::FLOAT / NULLIF(COUNT(*), 0) AS error_rate, COUNT(*) AS total
    FROM inference_logs WHERE timestamp > NOW() - INTERVAL '5 minutes'
"""

GET_PER_MODEL_STATS = """
    SELECT model_name, COUNT(*) AS total_requests, AVG(latency_ms) AS avg_latency, AVG(confidence) AS avg_confidence, COUNT(*) FILTER (WHERE status = 'error') AS error_count
    FROM inference_logs WHERE timestamp > NOW() - INTERVAL '5 minutes' GROUP BY model_name
"""

ACKNOWLEDGE_ALERT = """
    UPDATE alert_history SET acknowledged = TRUE, acknowledged_at = NOW(), acknowledged_by = $2 WHERE id = $1::uuid
"""
