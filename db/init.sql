CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS inference_logs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id      VARCHAR(64) NOT NULL UNIQUE,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(32) NOT NULL,
    input_shape     VARCHAR(64),
    input_hash      VARCHAR(64),
    prediction      VARCHAR(128) NOT NULL,
    predicted_class INTEGER NOT NULL,
    confidence      FLOAT NOT NULL,
    top_3_classes   JSONB,
    latency_ms      FLOAT NOT NULL,
    preprocess_ms   FLOAT,
    inference_ms    FLOAT,
    postprocess_ms  FLOAT,
    cpu_percent     FLOAT,
    memory_percent  FLOAT,
    runtime_type    VARCHAR(16) DEFAULT 'onnx',
    status          VARCHAR(16) DEFAULT 'success',
    error_message   TEXT,
    model_name      VARCHAR(64) DEFAULT 'fashion_mnist'
);

CREATE INDEX idx_inference_timestamp ON inference_logs(timestamp DESC);
CREATE INDEX idx_inference_model ON inference_logs(model_version);
CREATE INDEX idx_inference_model_name ON inference_logs(model_name);
CREATE INDEX idx_inference_status ON inference_logs(status);

CREATE TABLE IF NOT EXISTS drift_scores (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(32) NOT NULL,
    psi_score       FLOAT NOT NULL,
    kl_divergence   FLOAT NOT NULL,
    chi_square_stat FLOAT,
    chi_square_p    FLOAT,
    is_drifted      BOOLEAN NOT NULL DEFAULT FALSE,
    drift_severity  VARCHAR(16) DEFAULT 'none',
    sample_size     INTEGER NOT NULL,
    training_dist   JSONB,
    production_dist JSONB
);

CREATE INDEX idx_drift_timestamp ON drift_scores(timestamp DESC);

CREATE TABLE IF NOT EXISTS alert_history (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    alert_type      VARCHAR(64) NOT NULL,
    severity        VARCHAR(16) NOT NULL,
    message         TEXT NOT NULL,
    metric_value    FLOAT,
    threshold_value FLOAT,
    acknowledged    BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(128)
);

CREATE INDEX idx_alert_timestamp ON alert_history(timestamp DESC);
CREATE INDEX idx_alert_type ON alert_history(alert_type);
CREATE INDEX idx_alert_severity ON alert_history(severity);

CREATE TABLE IF NOT EXISTS model_registry (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version   VARCHAR(32) NOT NULL UNIQUE,
    model_type      VARCHAR(32) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_path       VARCHAR(256),
    file_size_bytes BIGINT,
    accuracy        FLOAT,
    num_classes     INTEGER,
    input_shape     VARCHAR(64),
    description     TEXT,
    is_active       BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS system_snapshots (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cpu_percent     FLOAT NOT NULL,
    memory_percent  FLOAT NOT NULL,
    memory_used_mb  FLOAT NOT NULL,
    disk_percent    FLOAT,
    active_connections INTEGER,
    requests_per_sec   FLOAT
);

CREATE INDEX idx_system_timestamp ON system_snapshots(timestamp DESC);

CREATE OR REPLACE VIEW hourly_metrics AS
SELECT
    date_trunc('hour', timestamp) AS hour,
    COUNT(*) AS total_requests,
    AVG(latency_ms) AS avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency,
    AVG(confidence) AS avg_confidence,
    MIN(confidence) AS min_confidence,
    COUNT(*) FILTER (WHERE status = 'error') AS error_count,
    COUNT(*) FILTER (WHERE status = 'error')::FLOAT / NULLIF(COUNT(*), 0) AS error_rate
FROM inference_logs
GROUP BY date_trunc('hour', timestamp)
ORDER BY hour DESC;

CREATE OR REPLACE VIEW model_performance AS
SELECT
    model_version,
    COUNT(*) AS total_predictions,
    AVG(latency_ms) AS avg_latency,
    AVG(confidence) AS avg_confidence,
    MIN(timestamp) AS first_seen,
    MAX(timestamp) AS last_seen
FROM inference_logs
GROUP BY model_version;
