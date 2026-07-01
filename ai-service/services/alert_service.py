import time
from datetime import datetime
from database.connection import db_pool
from database.redis_client import redis_manager
from database.queries import INSERT_ALERT
from config.settings import settings
from telemetry.custom_metrics import ALERTS_TOTAL

class AlertService:
    def __init__(self):
        self._last_alert_time = {}
        self._cooldown_seconds = 10

    async def check_and_alert(self, alert_type: str, metric_value: float, threshold: float, message: str, severity: str = "warning"):
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self._cooldown_seconds:
            return

        if metric_value > threshold:
            self._last_alert_time[alert_type] = now
            if metric_value > threshold * 2:
                severity = "critical"

            alert_data = {
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "metric_value": metric_value,
                "threshold": threshold,
                "timestamp": datetime.utcnow().isoformat()
            }

            try:
                await db_pool.execute(
                    INSERT_ALERT, alert_data["alert_type"], alert_data["severity"],
                    alert_data["message"], alert_data["metric_value"], alert_data["threshold"]
                )
                ALERTS_TOTAL.labels(alert_type=alert_data["alert_type"], severity=alert_data["severity"]).inc()
            except Exception as e: print(f"Alert DB store error: {e}")

            icon = {"info": "[INFO]", "warning": "[WARNING]", "critical": "[CRITICAL]"}.get(severity, "[ALERT]")
            print(f"{icon} {message}")

            await redis_manager.publish("alerts_stream", alert_data)

            if settings.alert_webhook_url:
                await self._send_webhook(alert_data)

            if settings.slack_webhook_url:
                await self._send_slack_alert(alert_data)

    async def _send_webhook(self, alert: dict):
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(settings.alert_webhook_url, json=alert)
        except Exception as e:
            print(f"webhook error: {e}")

    async def _send_slack_alert(self, alert: dict):
        try:
            import httpx
            import time
            icon = {"info": "[INFO]", "warning": "[WARNING]", "critical": "[CRITICAL]"}.get(alert["severity"], "[ALERT]")
            color = {"info": "#36a64f", "warning": "#ffcc00", "critical": "#ff002b"}.get(alert["severity"], "#cccccc")
            payload = {
                "attachments": [
                    {
                        "fallback": f"[{alert['severity'].upper()}] {alert['message']}",
                        "color": color,
                        "pretext": f"{icon} *Alert*",
                        "fields": [
                            {"title": "Alert Type", "value": alert["alert_type"], "short": True},
                            {"title": "Severity", "value": alert["severity"].upper(), "short": True},
                            {"title": "Message", "value": alert["message"], "short": False},
                            {"title": "Metric Value vs Threshold", "value": f"Current: `{alert['metric_value']:.4f}` | Threshold: `{alert['threshold']:.4f}`", "short": False}
                        ],
                        "footer": "aiobs",
                        "ts": int(time.time())
                    }
                ]
            }
            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(settings.slack_webhook_url, json=payload)
        except Exception as e:
            print(f"slack error: {e}")

    async def evaluate_prediction(self, latency_ms: float, confidence: float, error_rate: float):
        if latency_ms > 2000:
            await self.check_and_alert("high_latency", latency_ms, 2000, f"Inference latency exceeded 2s: {latency_ms:.0f}ms")
        if confidence < 0.3:
            await self.check_and_alert("low_confidence", 1.0 - confidence, 0.7, f"Very low prediction confidence: {confidence:.3f}")
        if error_rate > 0.1:
            await self.check_and_alert("error_spike", error_rate, 0.1, f"Error rate exceeded 10%: {error_rate*100:.1f}%", severity="critical")

alert_service = AlertService()
