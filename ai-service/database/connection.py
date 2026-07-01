import asyncpg
from typing import Optional, List, Dict, Any
from config.settings import settings

class DatabasePool:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                user=settings.postgres_user,
                password=settings.postgres_password,
                database=settings.postgres_db,
                host=settings.postgres_host,
                port=settings.postgres_port,
                min_size=2,
                max_size=10
            )
            if self.pool is not None:
                try:
                    async with self.pool.acquire() as conn:
                        await conn.execute("ALTER TABLE system_snapshots ADD COLUMN IF NOT EXISTS gpu_percent FLOAT;")
                        await conn.execute("ALTER TABLE system_snapshots ADD COLUMN IF NOT EXISTS gpu_memory_percent FLOAT;")
                        await conn.execute("ALTER TABLE system_snapshots ADD COLUMN IF NOT EXISTS gpu_temp FLOAT;")
                    print("Database migration successful.")
                except Exception as me:
                    print(f"Database migration warning: {me} (continuing anyway)")
            print(f"PostgreSQL connected: {settings.postgres_host}:{settings.postgres_port}")
        except Exception as e:
            print(f"PostgreSQL connection failed: {e}")
            self.pool = None

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("PostgreSQL disconnected.")

    async def execute(self, query: str, *args) -> str:
        if not self.pool: return "no_pool"
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        if not self.pool: return []
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        if not self.pool: return None
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        if not self.pool: return None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

db_pool = DatabasePool()
