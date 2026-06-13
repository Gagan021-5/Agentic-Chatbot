"""
═══════════════════════════════════════════════════════════════
Redis Client — Upstash REST-compatible session reader
═══════════════════════════════════════════════════════════════
Reads RentPrompts session keys (session:{id}).
Uses httpx for async HTTP-based Redis access via Upstash REST API.
"""

import json
import hashlib
import structlog
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class RedisClient:
    """Async Upstash Redis REST client matching the RentPrompts session format."""

    def __init__(self, url: str, token: str):
        self.base_url = url.rstrip("/")
        self.token = token
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=10.0,
        )

    async def close(self):
        await self._client.aclose()

    def _session_key(self, session_id: str) -> str:
        """Use RentPrompts key format: session:{id}"""
        return f"session:{session_id}"

    def _cache_key(self, prefix: str, query: str) -> str:
        """Generate a deterministic cache key."""
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"ai_cache:{prefix}:{h}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=3))
    async def get_session(self, session_id: str) -> dict | None:
        """Read a session from Upstash Redis using RentPrompts session keys."""
        try:
            key = self._session_key(session_id)
            resp = await self._client.get(f"/get/{key}")
            data = resp.json()
            result = data.get("result")
            if result is None:
                return None
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception as e:
            logger.error("redis_get_session_error", session_id=session_id, error=str(e))
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=3))
    async def set_cache(self, key: str, value: dict, ttl_seconds: int = 3600):
        """Write a cached value to Redis with TTL."""
        try:
            payload = json.dumps(value)
            resp = await self._client.post(
                "/",
                json=["SET", key, payload, "EX", str(ttl_seconds)],
            )
            return resp.json()
        except Exception as e:
            logger.error("redis_set_cache_error", key=key, error=str(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=0.5, max=3))
    async def get_cache(self, key: str) -> dict | None:
        """Read a cached value from Redis."""
        try:
            resp = await self._client.get(f"/get/{key}")
            data = resp.json()
            result = data.get("result")
            if result is None:
                return None
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception as e:
            logger.error("redis_get_cache_error", key=key, error=str(e))
            return None

    async def get_or_set_cache(self, prefix: str, query: str, ttl: int, factory):
        """Cache-aside pattern: return cached value or compute and store."""
        key = self._cache_key(prefix, query)
        cached = await self.get_cache(key)
        if cached is not None:
            return cached, True

        result = await factory()
        await self.set_cache(key, result, ttl)
        return result, False
