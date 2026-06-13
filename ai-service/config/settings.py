"""
RentPrompts Python Backend - Pydantic settings for the sole backend.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment."""

    # --- Gemini ---
    gemini_api_key: str = Field(default="", description="Google AI Studio API key for embeddings + LLM")

    # --- Groq ---
    groq_api_key: str = Field(default="", description="Groq API key for fast triage/extraction")

    # --- OpenRouter ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key for prompt generation")

    # --- Upstash Redis ---
    upstash_redis_rest_url: str = Field(default="", description="Upstash Redis REST endpoint")
    upstash_redis_rest_token: str = Field(default="", description="Upstash Redis REST auth token")

    # --- ChromaDB ---
    chroma_persist_dir: str = Field(default="./data/chromadb", description="ChromaDB persistence directory")

    # --- Payload CMS ---
    payload_cms_url: str = Field(default="http://localhost:3000", description="Payload CMS base URL")
    payload_cms_api_key: str = Field(default="", description="Payload CMS API key for auth")

    # --- Auth ---
    auth_secret_key: str = Field(
        default="rp-backend-secret-change-me-in-prod",
        description="Secret key for API key / JWT auth (React → Python)",
    )
    auth_mode: str = Field(default="none", description="Auth mode: none (dev), api_key, jwt")

    # --- CORS ---
    react_origin: str = Field(
        default="http://localhost:5173",
        description="Primary React dev/prod origin (also used in CORS allow list)",
    )
    cors_origins: str = Field(
        default="http://localhost:5173",
        description="Comma-separated allowed CORS origins",
    )

    # --- Murf TTS ---
    murf_api_key: str = Field(default="", description="Murf.ai TTS API key for audio preview")

    # --- Service ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")
    app_version: str = Field(default="1.0.0")

    # --- Embedding ---
    embedding_model: str = Field(default="models/gemini-embedding-001")
    embedding_dimensions: int = Field(default=768)

    # --- RAG ---
    rag_top_k: int = Field(default=5)
    rag_score_threshold: float = Field(default=0.3)

    # --- Redis ---
    cache_ttl_seconds: int = Field(default=3600)
    session_ttl_seconds: int = Field(default=1800)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def cors_origin_list(self) -> list[str]:
        origins = {o.strip() for o in self.cors_origins.split(",") if o.strip()}
        if self.react_origin.strip():
            origins.add(self.react_origin.strip())
        return sorted(origins)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
