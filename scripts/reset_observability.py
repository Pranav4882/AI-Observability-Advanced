import subprocess
import json
import sys
import uuid
from datetime import datetime, timedelta

def run_sql(sql_command: str) -> str:
    cmd = [
        "docker", "compose", "exec", "-T", "postgres", 
        "psql", "-U", "aiobs_user", "-d", "aiobs_db", "-c", sql_command
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing database command: {e.stderr}")
        sys.exit(1)

def run_redis(redis_command: str) -> str:
    cmd = [
        "docker", "compose", "exec", "-T", "redis", 
        "redis-cli", redis_command
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing Redis command: {e.stderr}")
        return ""

def hard_reset():
    print("Performing a hard reset on the observability database & cache...")
    truncate_sql = "TRUNCATE TABLE inference_logs, drift_scores, alert_history, system_snapshots CASCADE;"
    run_sql(truncate_sql)
    print("  PostgreSQL tables truncated successfully.")

    run_redis("FLUSHALL")
    print("  Redis cache flushed successfully.")
    print("\n[SUCCESS] Observability state has been completely reset to zero!")

def insert_balanced_logs():
    print("Generating balanced recovery logs (organically lowering drift score)...")
    
    fashion_classes = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    digit_classes = [str(d) for d in range(10)]
    sentiment_classes = ["negative", "positive"]
    
    now = datetime.utcnow()
    logs_to_insert = []
    
    for cls_idx, cls_name in enumerate(fashion_classes):
        for _ in range(5):
            req_id = str(uuid.uuid4())[:12]
            ts = now - timedelta(seconds=2 * (cls_idx * 5 + _))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            logs_to_insert.append(
                f"('{req_id}', 'v2.0.0', '(1,1,28,28)', '{cls_name}', {cls_idx}, 0.95, "
                f"'[]'::jsonb, 12.5, 2.1, 9.4, 1.0, 15.0, 45.0, 'onnx', 'success', NULL, 'fashion_mnist', '{ts_str}')"
            )

    for cls_idx, cls_name in enumerate(digit_classes):
        for _ in range(5):
            req_id = str(uuid.uuid4())[:12]
            ts = now - timedelta(seconds=2 * (cls_idx * 5 + _))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            logs_to_insert.append(
                f"('{req_id}', 'v1.0.0', '(1,1,28,28)', '{cls_name}', {cls_idx}, 0.98, "
                f"'[]'::jsonb, 8.2, 1.8, 5.9, 0.5, 12.0, 42.0, 'onnx', 'success', NULL, 'mnist_digits', '{ts_str}')"
            )

    for cls_idx, cls_name in enumerate(sentiment_classes):
        for _ in range(25):
            req_id = str(uuid.uuid4())[:12]
            ts = now - timedelta(seconds=4 * (cls_idx * 25 + _))
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
            logs_to_insert.append(
                f"('{req_id}', 'v1.0.0', '(1,512)', '{cls_name}', {cls_idx}, 0.99, "
                f"'[]'::jsonb, 4.5, 0.9, 3.2, 0.4, 8.0, 39.0, 'onnx', 'success', NULL, 'sentiment', '{ts_str}')"
            )

    values_str = ",\n    ".join(logs_to_insert)
    insert_sql = f"""
        INSERT INTO inference_logs (
            request_id, model_version, input_shape, prediction, predicted_class, confidence, 
            top_3_classes, latency_ms, preprocess_ms, inference_ms, postprocess_ms,
            cpu_percent, memory_percent, runtime_type, status, error_message, model_name, timestamp
        ) VALUES {values_str};
    """
    
    clear_recent_sql = "DELETE FROM inference_logs WHERE timestamp > NOW() - INTERVAL '5 minutes';"
    run_sql(clear_recent_sql)
    run_sql(insert_sql)
    print(f"  Successfully inserted 150 balanced logs in PostgreSQL.")
    
    run_redis("DEL requests:total")
    print("  Redis metrics cleared to sync with database.")
    print("\n[SUCCESS] Recovery traffic simulated! Run curl -X POST http://localhost:8000/drift/check to verify PSI returns to 0.0.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--hard":
        hard_reset()
    else:
        insert_balanced_logs()

if __name__ == "__main__":
    main()
