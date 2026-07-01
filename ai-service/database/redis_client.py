import redis.asyncio as aioredis
import json
from typing import Optional, Any
from config.settings import settings

class RedisManager:
    def __init__(self):
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        try:
            self.client = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,
                socket_connect_timeout=5
            )
            await self.client.ping()
            print(f"Redis connected: {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.client = None

    async def disconnect(self):
        if self.client:
            await self.client.close()
            print("Redis disconnected.")

    async def set_json(self, key: str, data: Any, ttl: int = 10):
        if self.client:
            await self.client.set(key, json.dumps(data), ex=ttl)

    async def get_json(self, key: str) -> Optional[Any]:
        if not self.client: return None
        val = await self.client.get(key)
        return json.loads(val) if val else None

    async def increment(self, key: str, ttl: int = 60) -> int:
        if not self.client: return 0
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl)
        results = await pipe.execute()
        return results[0]

    async def publish(self, channel: str, data: Any):
        if self.client:
            await self.client.publish(channel, json.dumps(data))

redis_manager = RedisManager()
