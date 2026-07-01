# AI Observability Platform

A real-time machine learning model monitoring and observability suite that logs inferences, tracks statistical data drift, alerts stakeholders, and monitors runtime system metrics using **FastAPI, PostgreSQL, Redis, Prometheus, and Grafana**.

---

## 🚀 Key Features

*   **Multi-Model Monitoring**: Out-of-the-box telemetry tracking for three active models:
    1.  `fashion_mnist` — Fashion template classifier (10 categories).
    2.  `mnist_digits` — Handwritten digits classifier (0–9).
    3.  `sentiment` — Sentiment NLP classifier (positive/negative reviews).
*   **Performance Telemetry**: Logs latencies (broken down by preprocessing, inference engine, and postprocessing runs) and prediction confidence.
*   **Automated Drift Detection**: Performs periodic class distribution calculations against training baselines using **Population Stability Index (PSI)**, **Kullback-Leibler (KL) Divergence**, and **Chi-Square** tests.
*   **System Metrics**: Tracks CPU, memory, and simulated GPU loads/temperatures.
*   **Alert Pipelines**: Archives warnings in PostgreSQL, streams alerts over WebSockets, and dispatches JSON updates to Slack webhooks.
*   **Interactive Testing UI**: Features drawing canvases, dropzones, and storage trackers to inspect the stack easily.

---

## 🗺️ System Architecture

```
User / Simulation Scripts
           │
           ▼  (REST / WebSockets)
  +-----------------------------------------------------------+
  |              FastAPI Gateway (ai-service:8000)            |
  +-----------------------------------------------------------+
       │                      │                      │
       │ (Logs & Alerts)      │ (Real-Time Streams)  │ (Prometheus Scrape)
       ▼                      ▼                      ▼
  +---------------+      +---------------+      +--------------+
  |  PostgreSQL   |      |  Redis Cache  |      |  Prometheus  |
  |  (Port 5432)  |      |  (Port 6379)  |      |  (Port 9090) |
  +---------------+      +---------------+      +--------------+
                                                       │
                                                       ▼ (Metrics Query)
                                                +--------------+
                                                |   Grafana    |
                                                |  (Port 3000) |
                                                +--------------+
```

### Port Mapping & Services

| Service Name | Docker Container / Image | Internal Port | Description |
|---|---|---|---|
| **ai-service** | Custom build (`python:3.12-slim`) | `8000` | FastAPI server, runs ONNX inference, computes drift, serves client UI. |
| **postgres** | `postgres:16-alpine` | `5432` | Relational store for inference logs, drift calculations, and alert logs. |
| **redis** | `redis:7-alpine` | `6379` | Key-value store for metrics snapshots and WebSockets Pub/Sub broker. |
| **prometheus** | `prom/prometheus:v2.54.1` | `9090` | Time-series database scraping `/metrics/prometheus` every 2s. |
| **grafana** | `grafana/grafana:11.3.0` | `3000` | Auto-provisioned dashboards dashboard metrics charts. |

---

## 🛠️ Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop) with **Docker Compose v2** running on the host machine.
*   [Python 3.10+](https://www.python.org/downloads/) (Only needed if running simulation scripts or training models locally).

---

## ⚡ Quick Start

### 1. Copy Environment Configuration
Create a local `.env` configuration file from the template:
```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 2. Launch the Application Stack
```bash
docker compose up --build -d
```
*Note: The first build will compile local Python dependencies and download base container images (takes ~3–5 mins).*

### 3. Check Liveness Status
Verify that the loader successfully bound the model weights and initialized the database pools:
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "service": "ai-observability",
  "models_loaded": ["fashion_mnist", "mnist_digits", "sentiment"]
}
```

### 4. Open Dashboards & UI
*   **Web UI Application Tester**: [http://localhost:8000/ui](http://localhost:8000/ui) (Live monitoring, drawing canvases, dropzones)
*   **API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Prometheus Console**: [http://localhost:9090](http://localhost:9090)
*   **Grafana Dashboard**: [http://localhost:3000](http://localhost:3000)
    *   *Credentials:* Username: `admin` | Password: `aiobs2024`

---

## ⚙️ Configuration (`.env`)

*   `DRIFT_PSI_THRESHOLD` (Default: `0.50`): The boundary beyond which a model class distribution is flagged as drifted.
*   `DRIFT_CHECK_INTERVAL` (Default: `5`): Polling speed (in seconds) for running background drift evaluations.
*   `SLACK_WEBHOOK_URL`: Slack Incoming Webhook URL to dispatch active channel alerts.

> 🔔 **Important: Setting up Slack Alerts**
> The template `.env.example` includes a placeholder Slack URL. To route alerts to your own Slack team channel:
> 1. Create an Incoming Webhook for your Slack workspace at [api.slack.com/messaging/webhooks](https://api.slack.com/messaging/webhooks).
> 2. Open your local `.env` file and replace the `SLACK_WEBHOOK_URL` value with your new webhook URL:
>    ```
>    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/OWN/WEBHOOK
>    ```
> 3. Restart the container to apply the changes:
>    ```bash
>    docker compose restart ai-service
>    ```

---

## 🔌 API Summary Reference

| HTTP Method | Route | Payload Details | Functionality |
|---|---|---|---|
| **POST** | `/predict/image` | File Upload (Grayscale PNG) | Evaluates Fashion MNIST classes (Defaults to `fashion_mnist` model). |
| **POST** | `/predict/image/digits` | File Upload (Grayscale PNG) | Custom route redirecting input to the `mnist_digits` model. |
| **POST** | `/predict/text` | JSON: `{"text": "string"}` | Evaluates text input reviews against the `sentiment` model. |
| **POST** | `/predict/batch` | Files List Upload (Max 32 files) | Evaluates multiple image payloads concurrently. |
| **GET** | `/predict/models` | None | Lists loaded metadata versions and labels for active models. |
| **GET** | `/health` | None | Basic liveness check route. |
| **GET** | `/readiness` | None | Returns liveness indicators for loaded models, Postgres connection, and Redis. |
| **GET** | `/db-redis-stats` | None | Returns database row counts and cached metrics key-value sizes. |
| **GET** | `/metrics/` | None | Summarizes current request counts, average latency, and cpu metrics. |
| **GET** | `/drift/status` | `?model=sentiment` (Optional) | Returns PSI scores and alert states. |
| **POST** | `/drift/check` | None | Forces execution of drift validation across all models immediately. |
| **GET** | `/alerts/` | None | Retrieves historic alert records from the database. |
| **POST** | `/alerts/acknowledge/{id}` | Path Parameter | Acknowledges an alert event in database logs. |
| **WS** | `/ws/live` | None | Real-time WebSocket connection to stream metrics and alerts. |

---

## 📊 Telemetry & Testing Scripts

The `scripts/` folder contains Python programs to inject metrics on the host machine. First, install the host requirements:
```bash
pip install httpx numpy Pillow
```

### 1. Organic Traffic Generator (`generate_traffic.py`)
Simulates regular requests distributed randomly across the image and text models:
```bash
# Sends 300 requests with a concurrency limit of 5 workers
python scripts/generate_traffic.py 300 5
```

### 2. Bias Drift Injector (`inject_drift.py`)
Submits imbalanced classes to skew distribution metrics and trigger warning panels:
```bash
# Sends 150 requests biased towards ankle boots (class 9) on fashion_mnist
python scripts/inject_drift.py 150 fashion_mnist 9

# Sends 100 requests biased towards positive reviews (class 1) on sentiment
python scripts/inject_drift.py 100 sentiment 1
```

### 3. Observability Reset Tool (`reset_observability.py`)
Resets metrics database logs back to baseline values:
```bash
# Option A: Generates organic balanced history entries to recover a high PSI score
python scripts/reset_observability.py

# Option B: Truncates database tables and cleans Redis keys completely
python scripts/reset_observability.py --hard
```

---

## 🧠 Model Training & datasets

The pre-trained weights (`.onnx`) and distributions are already packaged within the repository. To retrain models from scratch:

1.  Install training requirements (requires PyTorch):
    ```bash
    pip install -r ai-service/train-requirements.txt
    ```
2.  Run the training compiler:
    ```bash
    python ai-service/models/train.py
    ```
    *Note: The script automatically downloads datasets (MNIST, Fashion MNIST, and the 3,000 Yelp/Amazon/IMDb Sentiment reviews from the UCI Repository). If the system is offline, it falls back to a local sample list.*
3.  Restart the container:
    ```bash
    docker compose restart ai-service
    ```

---

## 🧼 Shutdown & Maintenance

*   **Stop Services (retrains volumes and metrics)**:
    ```bash
    docker compose down
    ```
*   **Stop and Wipe Database Volumes (Hard Reset)**:
    ```bash
    docker compose down -v
    ```
*   **Clean Up Cache / Reclaim Disk Space**:
    ```bash
    docker system prune -a --volumes
    ```
