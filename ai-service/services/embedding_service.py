"""
═══════════════════════════════════════════════════════════════
Embedding Service — Gemini text-embedding-004
═══════════════════════════════════════════════════════════════
Uses Google AI Studio API (not Vertex) for embeddings.
Matches the GEMINI_API_KEY already in the project .env.
"""

import structlog
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class EmbeddingService:
    """Generate embeddings using Gemini text-embedding-004 via AI Studio."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._document_embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.gemini_api_key,
            task_type="retrieval_document",
        )
        self._query_embeddings = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.gemini_api_key,
            task_type="retrieval_query",
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        return self._document_embeddings.embed_query(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string (uses retrieval_query task type for better recall)."""
        return self._query_embeddings.embed_query(query)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []

        results = []
        embeddings = self._query_embeddings if task_type == "retrieval_query" else self._document_embeddings

        for i in range(0, len(texts), 100):
            chunk = texts[i:i + 100]
            results.extend(embeddings.embed_documents(chunk))

        return results


# Singleton
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
