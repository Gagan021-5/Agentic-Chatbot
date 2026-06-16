"""
═══════════════════════════════════════════════════════════════
Session Service — Upstash Redis async session management
═══════════════════════════════════════════════════════════════
Upstash-backed session store. Key schema: session:{sessionId}
Uses the Upstash Redis REST API via httpx (no socket connection needed).
"""

import json
import time
from loguru import logger
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

settings = get_settings()

# In-memory fallback when Redis is unavailable
_memory_store: dict[str, dict] = {}
_memory_expiry: dict[str, float] = {}

SESSION_TTL = settings.session_ttl_seconds  # 1800s default (matches Node)


def _key(session_id: str) -> str:
    """Use RentPrompts key format exactly: session:{id}"""
    return f"session:{session_id}"


def _create_session(session_id: str) -> dict:
    """Create the default RentPrompts session shape."""
    return {
        "sessionId": session_id,
        "step": 0,
        "awaitingConfirmation": False,
        "confirmStep": None,
        "awaitingDeepAnswer": False,
        "currentDeepField": None,
        "deepAnswers": {},
        "appType": None,
        "modelId": None,
        "modelCost": None,
        "modelName": None,
        "extraction": None,
        "promptData": None,
        "scopeData": None,
        "seoData": None,
        "budgetPath": None,
        "history": [],
        "requirements": {},
        "currentField": None,
        "userType": "unknown",
        "enterpriseSignals": False,
        "lastQuestion": None,
        "isPivot": False,
        "dynamicContext": None,
        "formConfirmed": False,
        "triageRounds": 0,
        "awaitingTriageAnswer": False,
        "awaitingPromptTweak": False,
        "formatAskedByTriage": False,
        "formatConfirmedByUser": False,
        "domainIdentified": None,
        "languageMode": "English",
        "ragContext": None,
        "ragEnhancement": None,
        "modelGuidance": None,
    }


class SessionService:
    """Async Upstash Redis session manager with in-memory fallback."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.upstash_redis_rest_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.upstash_redis_rest_token}"},
            timeout=10.0,
        )
        self._has_redis = bool(
            settings.upstash_redis_rest_url
            and "your_" not in settings.upstash_redis_rest_url
            and settings.upstash_redis_rest_url.startswith("https://")
        )

    async def close(self):
        await self._client.aclose()

    async def get_session(self, session_id: str) -> dict | None:
        """Read session from Redis, fall back to memory."""
        if not self._has_redis:
            return self._get_memory(session_id)
        try:
            resp = await self._client.get(f"/get/{_key(session_id)}")
            data = resp.json()
            result = data.get("result")
            if result is None:
                return None
            if isinstance(result, str):
                return json.loads(result)
            return result
        except Exception as e:
            logger.warning(f"Redis get_session failed, using memory: {e}")
            return self._get_memory(session_id)

    async def get_or_create_session(self, session_id: str) -> dict:
        """Get existing session or create a new blank one."""
        existing = await self.get_session(session_id)
        if existing:
            return existing
        return _create_session(session_id)

    async def set_session(self, session_id: str, data: dict, ttl: int | None = None) -> None:
        """Write session by id with optional TTL override."""
        payload = {**data, "sessionId": session_id}
        ttl_seconds = ttl if ttl is not None else SESSION_TTL
        if not self._has_redis:
            self._set_memory(session_id, payload, ttl_seconds)
            return
        try:
            body = json.dumps(payload, default=str)
            await self._client.post(
                "/",
                json=["SET", _key(session_id), body, "EX", str(ttl_seconds)],
            )
        except Exception as e:
            logger.warning(f"Redis set_session failed, using memory: {e}")
            self._set_memory(session_id, payload, ttl_seconds)

    async def save_session(self, session: dict):
        """Write session to Redis with TTL, fall back to memory."""
        session_id = session.get("sessionId", "")
        if not self._has_redis:
            self._set_memory(session_id, session)
            return
        try:
            payload = json.dumps(session, default=str)
            await self._client.post(
                "/",
                json=["SET", _key(session_id), payload, "EX", str(SESSION_TTL)],
            )
        except Exception as e:
            logger.warning(f"Redis save_session failed, using memory: {e}")
            self._set_memory(session_id, session)

    async def delete_session(self, session_id: str):
        """Delete session from Redis."""
        if not self._has_redis:
            _memory_store.pop(_key(session_id), None)
            _memory_expiry.pop(_key(session_id), None)
            return
        try:
            await self._client.post("/", json=["DEL", _key(session_id)])
        except Exception as e:
            logger.warning(f"Redis delete_session failed: {e}")
            _memory_store.pop(_key(session_id), None)

    async def reset_app_specific_context(self, session_id: str) -> dict:
        """Reset app-specific context while preserving basic session identity."""
        session = await self.get_session(session_id)
        if not session:
            session = _create_session(session_id)
            
        session["step"] = 0
        session["awaitingConfirmation"] = False
        session["confirmStep"] = None
        session["awaitingDeepAnswer"] = False
        session["currentDeepField"] = None
        session["deepAnswers"] = {}
        session["appType"] = None
        session["modelId"] = None
        session["modelCost"] = None
        session["modelName"] = None
        session["extraction"] = None
        session["promptData"] = None
        session["scopeData"] = None
        session["seoData"] = None
        session["budgetPath"] = None
        session["requirements"] = {}
        session["currentField"] = None
        session["lastQuestion"] = None
        session["isPivot"] = True
        session["dynamicContext"] = None
        session["formConfirmed"] = False
        session["triageRounds"] = 0
        session["awaitingTriageAnswer"] = False
        session["awaitingPromptTweak"] = False
        session["formatAskedByTriage"] = False
        session["formatConfirmedByUser"] = False
        session["domainIdentified"] = None
        session["ragContext"] = None
        session["ragEnhancement"] = None
        session["modelGuidance"] = None
        
        await self.save_session(session)
        return session

    # --- Cache helpers (for RAG/web research caching) ---

    async def get_cache(self, key: str) -> dict | None:
        if not self._has_redis:
            return self._get_memory(key)
        try:
            resp = await self._client.get(f"/get/{key}")
            data = resp.json()
            result = data.get("result")
            if result is None:
                return None
            return json.loads(result) if isinstance(result, str) else result
        except Exception:
            return None

    async def set_cache(self, key: str, value: dict, ttl: int = 3600):
        if not self._has_redis:
            return
        try:
            await self._client.post(
                "/",
                json=["SET", key, json.dumps(value, default=str), "EX", str(ttl)],
            )
        except Exception as e:
            logger.warning(f"Redis set_cache failed: {e}")

    # --- Memory fallback ---

    def _get_memory(self, key: str) -> dict | None:
        k = key if key.startswith("session:") else _key(key)
        entry = _memory_store.get(k)
        if not entry:
            return None
        if _memory_expiry.get(k, 0) < time.time():
            _memory_store.pop(k, None)
            _memory_expiry.pop(k, None)
            return None
        return json.loads(json.dumps(entry))  # deep clone

    def _set_memory(self, session_id: str, session: dict, ttl: int | None = None):
        k = _key(session_id)
        _memory_store[k] = json.loads(json.dumps(session, default=str))
        _memory_expiry[k] = time.time() + (ttl if ttl is not None else SESSION_TTL)
