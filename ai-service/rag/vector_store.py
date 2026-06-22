"""
═══════════════════════════════════════════════════════════════
Vector Store Manager — ChromaDB with Gemini Embeddings
═══════════════════════════════════════════════════════════════
Manages collections, document ingestion, and similarity search.
"""

import os
import json
import structlog
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import Settings
from services.embedding_service import get_embedding_service

logger = structlog.get_logger(__name__)

# Knowledge base categories mapping to ChromaDB collections
COLLECTION_NAMES = {
    "models": "kb_models",
    "prompting": "kb_prompting",
    "examples": "kb_examples",
    "seo": "kb_seo",
    "marketplace": "kb_marketplace",
    "blueprints": "kb_blueprints",
}


class VectorStoreManager:
    """ChromaDB vector store with Gemini embeddings for RAG."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_service = get_embedding_service()
        self._client: chromadb.ClientAPI | None = None
        self._collections: dict[str, chromadb.Collection] = {}

    async def initialize(self):
        """Initialize ChromaDB client and collections."""
        try:
            # Try persistent client first (local development)
            persist_dir = self.settings.chroma_persist_dir
            os.makedirs(persist_dir, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Create/get all collections
            for category, collection_name in COLLECTION_NAMES.items():
                self._collections[category] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"},
                )

            logger.info(
                "chromadb_initialized",
                mode="persistent",
                path=persist_dir,
                collections=list(COLLECTION_NAMES.values()),
            )
        except Exception as e:
            logger.error("chromadb_init_error", error=str(e))
            raise

    def get_collection(self, category: str) -> chromadb.Collection | None:
        return self._collections.get(category)

    async def add_documents(
        self,
        category: str,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ):
        """Add documents to a specific knowledge base collection."""
        collection = self.get_collection(category)
        if not collection:
            logger.warning("collection_not_found", category=category)
            return

        if not documents:
            return

        # Generate embeddings
        embeddings = self.embedding_service.embed_batch(documents)

        # Generate IDs if not provided
        if ids is None:
            import hashlib
            ids = [
                f"{category}_{hashlib.sha256(doc.encode()).hexdigest()[:12]}"
                for doc in documents
            ]

        # Default metadata
        if metadatas is None:
            metadatas = [{"category": category, "source": "manual"}] * len(documents)

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("documents_added", category=category, count=len(documents))

    async def search(
        self,
        query: str,
        categories: list[str] | None = None,
        top_k: int = 5,
        metadata_filter: dict | None = None,
        boost_gold_standards: bool = False,
    ) -> list[dict]:
        """Search across one or more knowledge base collections."""
        if categories is None:
            categories = list(COLLECTION_NAMES.keys())

        query_embedding = self.embedding_service.embed_query(query)
        all_results = []

        for category in categories:
            collection = self.get_collection(category)
            if not collection:
                continue

            try:
                where_filter = metadata_filter if metadata_filter else None
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, 10),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )

                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        distance = results["distances"][0][i] if results.get("distances") else 1.0
                        # ChromaDB cosine distance: 0 = identical, 2 = opposite
                        # Convert to similarity score: 1 - (distance / 2)
                        similarity = 1.0 - (distance / 2.0)

                        metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                        relevance = round(similarity, 4)
                        if boost_gold_standards and metadata.get("source") == "marketplace_gold_standards.md":
                            relevance = min(1.0, relevance + 0.2)

                        all_results.append({
                            "content": doc,
                            "source": metadata.get("source", category),
                            "category": category,
                            "relevance_score": relevance,
                            "metadata": metadata,
                        })

            except Exception as e:
                logger.error("search_error", category=category, error=str(e))

        # Sort by relevance and take top_k
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_results[:top_k]

    async def search_with_filter(
        self,
        query: str,
        category: str,
        model_id: str | None = None,
        app_type: str | None = None,
        top_k: int = 5,
        boost_gold_standards: bool = False,
    ) -> list[dict]:
        """Search with metadata filtering for model-specific or type-specific docs."""
        metadata_filter = {}
        if model_id:
            metadata_filter["model_id"] = model_id
        if app_type:
            metadata_filter["app_type"] = app_type

        where = metadata_filter if metadata_filter else None
        return await self.search(
            query=query,
            categories=[category],
            top_k=top_k,
            metadata_filter=where,
            boost_gold_standards=boost_gold_standards,
        )

    def get_collection_stats(self) -> dict[str, int]:
        """Return document counts per collection."""
        stats = {}
        for category, collection in self._collections.items():
            try:
                stats[category] = collection.count()
            except Exception:
                stats[category] = -1
        return stats
