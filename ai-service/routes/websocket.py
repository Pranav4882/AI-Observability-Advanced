import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Set

router = APIRouter()
connected_clients: Set[WebSocket] = set()

@router.websocket("/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"WebSocket client connected. Total: {len(connected_clients)}")

    try:
        from database.redis_client import redis_manager
        if redis_manager.client:
            pubsub = redis_manager.client.pubsub()
            await pubsub.subscribe("metrics_stream", "alerts_stream")

            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    await websocket.send_json(data)

                try:
                    client_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                    if client_msg == "ping":
                        await websocket.send_json({"type": "pong"})
                except asyncio.TimeoutError:
                    pass
        else:
            from services.metrics_service import metrics_collector
            while True:
                summary = await metrics_collector.get_summary()
                await websocket.send_json({"type": "metrics_summary", **summary})
                await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        connected_clients.discard(websocket)
        print(f"WebSocket client disconnected. Total: {len(connected_clients)}")
