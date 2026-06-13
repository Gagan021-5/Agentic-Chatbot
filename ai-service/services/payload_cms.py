"""
═══════════════════════════════════════════════════════════════
Payload CMS Client — async REST client for all CMS operations
═══════════════════════════════════════════════════════════════
Centralizes Payload CMS calls.
- Model discovery (GET /api/models/modelType/{type})
- Rapp registration (POST /api/rapps, /api/privateRapps)
- Media upload (POST /api/media)
"""

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings

settings = get_settings()


class PayloadCMSClient:
    """Async Payload CMS REST client with retry logic."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.payload_cms_url.rstrip("/"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {settings.payload_cms_api_key}"}
                   if settings.payload_cms_api_key else {}),
            },
            timeout=30.0,
        )

    async def close(self):
        await self._client.aclose()

    # ─── Model Discovery ────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=0.5, max=5),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def get_models_by_type(self, model_type: str) -> list[dict]:
        """GET /api/models/modelType/{type} — discover available models."""
        try:
            resp = await self._client.get(f"/api/models/modelType/{model_type}")
            resp.raise_for_status()
            data = resp.json()
            # Payload CMS returns { docs: [...] } or the array directly
            if isinstance(data, dict) and "docs" in data:
                return data["docs"]
            if isinstance(data, list):
                return data
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"CMS get_models_by_type failed: HTTP {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"CMS get_models_by_type error: {e}")
            return []

    # ─── Rapp Registration ──────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def create_rapp(self, payload: dict, private: bool = False) -> dict:
        """POST /api/rapps or /api/privateRapps — register a new app."""
        endpoint = "/api/privateRapps" if private else "/api/rapps"
        try:
            resp = await self._client.post(endpoint, json=payload)
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"CMS rapp created: {result.get('id', 'unknown')} (private={private})")
            return result
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500]
            logger.error(f"CMS create_rapp failed: HTTP {e.response.status_code} — {body}")
            raise
        except Exception as e:
            logger.error(f"CMS create_rapp error: {e}")
            raise

    # ─── Media Upload ───────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def upload_media(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str = "image/png",
    ) -> dict:
        """POST /api/media — multipart file upload for preview images."""
        try:
            # Multipart upload — Payload CMS expects 'file' field
            files = {"file": (filename, file_bytes, content_type)}
            # Remove Content-Type header for multipart (httpx sets it with boundary)
            headers = dict(self._client.headers)
            headers.pop("Content-Type", None)

            resp = await self._client.post(
                "/api/media",
                files=files,
                headers=headers,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(f"CMS media uploaded: {result.get('id', 'unknown')}, filename={filename}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"CMS upload_media failed: HTTP {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"CMS upload_media error: {e}")
            raise

    # ─── Health Check ───────────────────────────────────────

    async def health_check(self) -> bool:
        """Check if Payload CMS is reachable."""
        try:
            resp = await self._client.get("/api/globals", timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False
