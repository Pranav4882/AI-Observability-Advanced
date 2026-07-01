import uuid
import time
import json
import psutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import List, Optional
from models.model_loader import model_manager
from utils.image_processing import preprocess_image, preprocess_text
from database.connection import db_pool
from database.redis_client import redis_manager
from database.queries import INSERT_INFERENCE_LOG
from config.settings import settings
from telemetry.custom_metrics import PREDICTION_CONFIDENCE, PREDICTIONS_PER_CLASS, PREDICTION_ANOMALY_SCORE, ALERTS_TOTAL

router = APIRouter()

async def _log_and_respond(req_id, model, shape, p_class, name, conf, top3, total_ms, pre_ms, inf_ms, runtime, input_anomaly=0.0):
    post_ms = total_ms - pre_ms - inf_ms
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    m_obj = model_manager.get(model)
    version = m_obj.model_version if m_obj else settings.model_version
    anomaly_score = 0.5 * (1.0 - conf) + 0.5 * input_anomaly

    PREDICTION_CONFIDENCE.labels(model_name=model).observe(conf)
    PREDICTIONS_PER_CLASS.labels(model_name=model, class_name=name, class_id=str(p_class)).inc()
    PREDICTION_ANOMALY_SCORE.labels(model_name=model).observe(anomaly_score)

    try:
        await db_pool.execute(
            INSERT_INFERENCE_LOG, req_id, version, shape, name, p_class, conf,
            json.dumps(top3), total_ms, pre_ms, inf_ms, post_ms, cpu, mem, runtime, "success", None, model
        )
    except Exception as e:
        print(f"DB log failed: {e}")

    try:
        await redis_manager.increment("requests:total")
        await redis_manager.increment(f"requests:{model}:class:{p_class}")
        payload = {
            "request_id": req_id, 
            "model": model, 
            "prediction": name, 
            "confidence": conf, 
            "anomaly_score": round(anomaly_score, 4),
            "latency_ms": round(total_ms, 2)
        }
        await redis_manager.set_json("latest_prediction", payload)
        await redis_manager.publish("metrics_stream", {"type": "prediction", "cpu_percent": cpu, "memory_percent": mem, **payload})
    except Exception:
        pass

    return {
        "request_id": req_id, "model": model, "prediction": name, "predicted_class": p_class,
        "confidence": round(conf, 4), "anomaly_score": round(anomaly_score, 4), "top_3": top3,
        "latency": {"total_ms": round(total_ms, 2), "preprocess_ms": round(pre_ms, 2), "inference_ms": round(inf_ms, 2), "postprocess_ms": round(post_ms, 2)},
        "model_info": {"version": version, "runtime": runtime},
        "system": {"cpu_percent": cpu, "memory_percent": mem}
    }

@router.post("/image")
async def predict_image(file: UploadFile = File(...), model: str = Query("fashion_mnist")):
    req_id = str(uuid.uuid4())[:12]
    t_start = time.perf_counter()
    try:
        img = await file.read()
        tensor, _, pre_ms = preprocess_image(img, model)
        
        import numpy as np
        mean_val = float(np.mean(tensor))
        input_anomaly = min(1.0, abs(mean_val) / 2.0)

        t0 = time.perf_counter()
        pcls, cname, conf, top3 = model_manager.predict(tensor, model)
        inf_ms = (time.perf_counter() - t0) * 1000
        tot_ms = (time.perf_counter() - t_start) * 1000
        m = model_manager.get(model)
        return await _log_and_respond(req_id, model, "(1,1,28,28)", pcls, cname, conf, top3, tot_ms, pre_ms, inf_ms, m.runtime_type if m else "unknown", input_anomaly=input_anomaly)
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/image/digits")
async def predict_digits(file: UploadFile = File(...)):
    return await predict_image(file, "mnist_digits")

@router.post("/text")
async def predict_text(text: str = Body(..., embed=True)):
    req_id = str(uuid.uuid4())[:12]
    t_start = time.perf_counter()
    try:
        tensor, _, pre_ms = preprocess_text(text)

        import numpy as np
        vocab_matches = float(np.sum(tensor > 0))
        input_anomaly = 1.0 if vocab_matches == 0 else max(0.0, 1.0 - (vocab_matches / max(1, len(text.split()))))

        t0 = time.perf_counter()
        pcls, cname, conf, top3 = model_manager.predict(tensor, "sentiment")
        inf_ms = (time.perf_counter() - t0) * 1000
        tot_ms = (time.perf_counter() - t_start) * 1000
        m = model_manager.get("sentiment")
        return await _log_and_respond(req_id, "sentiment", f"(1,{tensor.shape[1]})", pcls, cname, conf, top3, tot_ms, pre_ms, inf_ms, m.runtime_type if m else "unknown", input_anomaly=input_anomaly)
    except Exception as e:
        raise HTTPException(500, f"prediction error: {e}")

@router.post("/batch")
async def predict_batch(files: List[UploadFile] = File(...), model: str = Query("fashion_mnist")):
    results = []
    for f in files[:32]:
        try:
            img = await f.read()
            tensor, _, pre_ms = preprocess_image(img, model)
            t0 = time.perf_counter()
            pcls, cname, conf, _ = model_manager.predict(tensor, model)
            results.append({
                "filename": f.filename, "prediction": cname, "confidence": round(conf, 4),
                "latency_ms": round(((time.perf_counter() - t0)*1000) + pre_ms, 2)
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})
    return {"batch_size": len(files), "model": model, "results": results}

@router.get("/models")
async def list_models():
    return {"models": model_manager.list_models()}
