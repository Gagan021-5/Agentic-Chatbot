"""
═══════════════════════════════════════════════════════════════
Embedding Service — Gemini text-embedding-004
═══════════════════════════════════════════════════════════════
Uses Google AI Studio API (not Vertex) for embeddings.
Matches the GEMINI_API_KEY already in the project .env.
"""

import structlog
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# Configure Gemini with AI Studio key
genai.configure(api_key=settings.gemini_api_key)


class EmbeddingService:
    """Generate embeddings using Gemini text-embedding-004 via AI Studio."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document",
            output_dimensionality=self.dimensions,
        )
        return result["embedding"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string (uses retrieval_query task type for better recall)."""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=self.dimensions,
        )
        return result["embedding"]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        # Gemini supports batch embedding
        results = []
        # Process in chunks of 100
        for i in range(0, len(texts), 100):
            chunk = texts[i:i + 100]
            batch_result = genai.embed_content(
                model=self.model_name,
                content=chunk,
                task_type=task_type,
                output_dimensionality=self.dimensions,
            )
            results.extend(batch_result["embedding"])

        return results


# Singleton
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
