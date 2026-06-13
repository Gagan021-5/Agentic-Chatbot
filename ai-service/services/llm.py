"""
═══════════════════════════════════════════════════════════════
LLM Service — Groq + OpenRouter + Gemini async clients
═══════════════════════════════════════════════════════════════
Centralizes Gemini, Groq, and OpenRouter LLM calls.
"""

import json
import re
import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

settings = get_settings()


class LLMService:
    """Unified async LLM client for Groq, OpenRouter, and Gemini."""

    def __init__(self):
        self._groq_client = httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {settings.groq_api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        ) if settings.groq_api_key else None

        self._openrouter_client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5173",
                "X-Title": "RentPrompts Agent",
            },
            timeout=60.0,
        ) if settings.openrouter_api_key else None

    async def close(self):
        if self._groq_client:
            await self._groq_client.aclose()
        if self._openrouter_client:
            await self._openrouter_client.aclose()

    # ─── Groq (fast triage / extraction / preview text) ─────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=0.5, max=3))
    async def groq_completion(
        self,
        messages: list[dict],
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 500,
        temperature: float = 0.3,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
    ) -> dict:
        """Call Groq with a full messages array (extraction, triage, orchestration)."""
        if not self._groq_client:
            raise RuntimeError("GROQ_API_KEY not configured")

        body: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        resp = await self._groq_client.post("/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def groq_chat(
        self,
        system_prompt: str,
        user_content: str,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 500,
        temperature: float = 0.3,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Call Groq for fast inference (triage, extraction, preview)."""
        tc = tool_choice
        if isinstance(tool_choice, str):
            tc = tool_choice
        return await self.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=response_format,
            tools=tools,
            tool_choice=tc,
        )

    async def openrouter_completion(
        self,
        messages: list[dict],
        model: str = "meta-llama/llama-3.3-70b-instruct",
        max_tokens: int = 3000,
        temperature: float = 0.7,
        response_format: dict | None = None,
    ) -> str:
        """Call OpenRouter with a full messages array."""
        if not self._openrouter_client:
            raise RuntimeError("OPENROUTER_API_KEY not configured")

        body: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        resp = await self._openrouter_client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def groq_tool_call(
        self,
        system_prompt: str,
        user_content: str,
        tools: list[dict],
        model: str = "llama-3.3-70b-versatile",
    ) -> dict | None:
        """Call Groq with function calling and extract tool arguments."""
        try:
            result = await self.groq_chat(
                system_prompt=system_prompt,
                user_content=user_content,
                model=model,
                max_tokens=800,
                temperature=0.1,
                tools=tools,
                tool_choice="auto",
            )
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            # Tool call response
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
                return json.loads(args_str)

            # Sometimes model responds in content instead of tool_calls
            content = message.get("content", "")
            if content:
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    return json.loads(json_match.group())

            return None
        except Exception as e:
            logger.warning(f"Groq tool call failed: {e}")
            return None

    # ─── OpenRouter (prompt generation, SEO, test-prompt) ───

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    async def openrouter_chat(
        self,
        system_prompt: str,
        user_content: str,
        model: str = "google/gemini-1.5-flash",
        max_tokens: int = 3000,
        temperature: float = 0.7,
    ) -> str:
        """Call OpenRouter and return raw text response."""
        if not self._openrouter_client:
            raise RuntimeError("OPENROUTER_API_KEY not configured")

        resp = await self._openrouter_client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def openrouter_json(
        self,
        system_prompt: str,
        user_content: str,
        model: str = "google/gemini-1.5-flash",
        fallback_model: str = "meta-llama/llama-3.3-70b-instruct",
    ) -> dict:
        """Call OpenRouter, extract JSON from response. Auto-fallback on 429."""
        models = [model, fallback_model]

        for m in models:
            try:
                raw = await self.openrouter_chat(system_prompt, user_content, model=m)
                json_match = re.search(r"\{[\s\S]*\}", raw)
                if not json_match:
                    raise ValueError("No JSON found in response")

                # Sanitize control characters
                sanitized = json_match.group().replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                try:
                    return json.loads(sanitized)
                except json.JSONDecodeError:
                    # Last resort: strip all control chars
                    return json.loads(re.sub(r"[\x00-\x1f\x7f]", " ", json_match.group()))

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and m != models[-1]:
                    logger.warning(f"OpenRouter {m} rate limited, trying {models[-1]}")
                    continue
                raise
            except Exception as e:
                if m != models[-1]:
                    logger.warning(f"OpenRouter {m} failed: {e}, trying fallback")
                    continue
                raise

        raise RuntimeError("All OpenRouter models failed")

    # ─── Groq Vision (for image preview pipeline) ───────────

    async def groq_vision(
        self,
        image_data_url: str,
        prompt: str,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens: int = 280,
    ) -> str:
        """Call Groq Vision for image analysis (preview pipeline)."""
        if not self._groq_client:
            raise RuntimeError("GROQ_API_KEY not configured")

        resp = await self._groq_client.post(
            "/chat/completions",
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    @property
    def has_groq(self) -> bool:
        return self._groq_client is not None

    @property
    def has_openrouter(self) -> bool:
        return self._openrouter_client is not None
