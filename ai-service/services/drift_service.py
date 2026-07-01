import asyncio
import numpy as np
import json
import math
import time
import os
from typing import Dict, Tuple, Optional, Set

from database.connection import db_pool
from database.redis_client import redis_manager
from database.queries import INSERT_DRIFT_SCORE, GET_RECENT_PREDICTIONS_BY_MODEL
from models.model_loader import model_manager
from config.settings import settings
from telemetry.custom_metrics import DRIFT_PSI_SCORE, DRIFT_KL_DIVERGENCE, DRIFT_IS_DETECTED, PREDICTION_CLASS_IMBALANCE

def _igf(a, x):
    if x < 0.0 or a <= 0.0:
        return 0.0
    sum_val = 1.0 / a
    temp = sum_val
    n = 1
    while True:
        temp = temp * x / (a + n)
        sum_val += temp
        if abs(temp) < abs(sum_val) * 1e-15:
            break
        n += 1
    return sum_val * math.exp(-x + a * math.log(x))

def _chi2_sf(x, df):
    if x <= 0:
        return 1.0
    try:
        val = 1.0 - (_igf(df / 2.0, x / 2.0) / math.gamma(df / 2.0))
        return max(0.0, min(1.0, val))
    except Exception:
        return 1.0

class DriftService:
    def __init__(self):
        self.latest_results: Dict[str, Dict] = {}
        self.last_checked_predictions: Dict[str, tuple] = {}

    async def periodic_drift_check(self):
        try:
            for name in model_manager.models:
                DRIFT_PSI_SCORE.labels(model_name=name).set(0.0)
                DRIFT_KL_DIVERGENCE.labels(model_name=name).set(0.0)
                DRIFT_IS_DETECTED.labels(model_name=name).set(0)
        except Exception as e:
            print(f"drift gauge init error: {e}")

        while True:
            try:
                await asyncio.sleep(settings.drift_check_interval)
                for name, model in model_manager.models.items():
                    if model.training_baseline is None:
                        continue
                    result = await self.run_drift_check(name)
                    if result:
                        self.latest_results[name] = result
                        await db_pool.execute(
                            INSERT_DRIFT_SCORE, model.model_version, result["psi_score"], result["kl_divergence"],
                            result["chi_square_stat"], result["chi_square_p"], result["is_drifted"],
                            result["drift_severity"], result["sample_size"],
                            json.dumps(result["training_dist"]), json.dumps(result["production_dist"])
                        )
                        DRIFT_PSI_SCORE.labels(model_name=name).set(result["psi_score"])
                        DRIFT_KL_DIVERGENCE.labels(model_name=name).set(result["kl_divergence"])
                        DRIFT_IS_DETECTED.labels(model_name=name).set(1 if result["is_drifted"] else 0)
                        await redis_manager.publish("metrics_stream", {"type": "drift_update", "model": name, **result})
                        
                        if result["is_drifted"]:
                            from services.alert_service import alert_service
                            await alert_service.check_and_alert(
                                f"drift_{name}", result["psi_score"], settings.drift_psi_threshold,
                                f"Drift detected on [{name}]! PSI={result['psi_score']:.3f}, Severity={result['drift_severity']}"
                            )
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"drift check: {e}")
                await asyncio.sleep(10)

    async def run_drift_check(self, model_name: str, force: bool = False) -> Optional[Dict]:
        model = model_manager.get(model_name)
        if not model or not model.training_baseline:
            return None

        baseline = model.training_baseline
        num_classes = baseline.get("num_classes", len(baseline["class_distribution"]))

        rows = await db_pool.fetch(GET_RECENT_PREDICTIONS_BY_MODEL, model_name)
        if len(rows) < 10:
            return None

        current_state = (len(rows), rows[0]["timestamp"])
        if not force and self.last_checked_predictions.get(model_name) == current_state:
            return None
        self.last_checked_predictions[model_name] = current_state

        class_counts = np.zeros(num_classes)
        confidences = []
        for row in rows:
            idx = row["predicted_class"]
            if 0 <= idx < num_classes:
                class_counts[idx] += 1
            confidences.append(row["confidence"])

        total = class_counts.sum()
        if total == 0:
            return None
        prod_dist = class_counts / total
        train_dist = np.array(baseline["class_distribution"])

        psi = self._calculate_psi(train_dist, prod_dist)
        kl = self._calculate_kl_divergence(train_dist, prod_dist)
        chi2_stat, chi2_p = self._calculate_chi_square(train_dist * len(rows), class_counts)

        eps = 1e-9
        entropy = -np.sum(prod_dist * np.log(prod_dist + eps))
        max_entropy = np.log(len(prod_dist))
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 1.0

        is_drifted = psi > settings.drift_psi_threshold
        severity = "none"
        if psi >= 0.3: severity = "high"
        elif psi >= 0.2: severity = "medium"
        elif psi >= 0.1: severity = "low"

        DRIFT_PSI_SCORE.labels(model_name=model_name).set(round(float(psi), 4))
        DRIFT_KL_DIVERGENCE.labels(model_name=model_name).set(round(float(kl), 4))
        DRIFT_IS_DETECTED.labels(model_name=model_name).set(1 if is_drifted else 0)
        PREDICTION_CLASS_IMBALANCE.labels(model_name=model_name).set(round(norm_entropy, 4))

        return {
            "model": model_name, "psi_score": round(float(psi), 4), "kl_divergence": round(float(kl), 4),
            "chi_square_stat": round(float(chi2_stat), 4), "chi_square_p": round(float(chi2_p), 4),
            "is_drifted": is_drifted, "drift_severity": severity, "sample_size": len(rows),
            "training_dist": train_dist.tolist(), "production_dist": prod_dist.tolist(),
            "avg_confidence": round(float(np.mean(confidences)), 4),
        }

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        eps = 1e-6
        expected = np.clip(expected, eps, None)
        actual = np.clip(actual, eps, None)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    def _calculate_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        eps = 1e-6
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        p, q = p / p.sum(), q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def _calculate_chi_square(self, expected_counts: np.ndarray, observed_counts: np.ndarray) -> Tuple[float, float]:
        try:
            expected_counts = np.clip(expected_counts, 1, None)
            chi2_stat = float(np.sum((observed_counts - expected_counts) ** 2 / expected_counts))
            df = len(expected_counts) - 1
            pval = _chi2_sf(chi2_stat, df)
            return chi2_stat, pval
        except Exception:
            return 0.0, 1.0

    async def get_status(self, model_name: Optional[str] = None) -> Dict:
        if model_name:
            r = self.latest_results.get(model_name)
            return r if r else {
                "model": model_name, "psi_score": 0, "kl_divergence": 0,
                "is_drifted": False, "drift_severity": "none", "message": "not enough data"
            }
        return {
            "models": {
                name: self.latest_results.get(name, {"message": "no data"})
                for name in model_manager.models
            }
        }

drift_service = DriftService()
