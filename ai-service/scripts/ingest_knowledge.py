"""
═══════════════════════════════════════════════════════════════
Knowledge Base Ingestion Script
═══════════════════════════════════════════════════════════════
Reads markdown files from knowledge/ and ingests into ChromaDB.
Run: python scripts/ingest_knowledge.py
"""

import os
import sys
import asyncio

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from rag.vector_store import VectorStoreManager
from rag.chunker import chunk_markdown

settings = get_settings()

# Category → directory mapping
KNOWLEDGE_DIRS = {
    "models": "knowledge/models",
    "prompting": "knowledge/prompting",
    "examples": "knowledge/examples",
    "seo": "knowledge/seo",
    "marketplace": "knowledge/marketplace",
}


async def ingest_all():
    """Ingest all knowledge base documents into ChromaDB."""
    print("═══════════════════════════════════════════")
    print("  RentPrompts Knowledge Base Ingestion")
    print("═══════════════════════════════════════════")

    # Initialize vector store
    vs = VectorStoreManager(settings)
    await vs.initialize()

    total_docs = 0

    for category, dir_path in KNOWLEDGE_DIRS.items():
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dir_path)

        if not os.path.exists(full_path):
            print(f"  ⚠️  Skipping {category}: directory not found at {full_path}")
            continue

        print(f"\n  📂 Processing: {category}/")

        for filename in os.listdir(full_path):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(full_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Chunk the document
            chunks = chunk_markdown(content, chunk_size=512)

            if not chunks:
                print(f"     ⏭️  {filename}: no chunks extracted")
                continue

            # Prepare documents and metadata
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [
                {
                    "category": category,
                    "source": filename,
                    "header": chunk.get("header", ""),
                    "level": chunk.get("level", 0),
                }
                for chunk in chunks
            ]
            ids = [
                f"{category}_{filename}_{i}"
                for i in range(len(documents))
            ]

            # Ingest
            await vs.add_documents(
                category=category,
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

            total_docs += len(documents)
            print(f"     ✅ {filename}: {len(documents)} chunks ingested")

    # Print stats
    print("\n═══════════════════════════════════════════")
    stats = vs.get_collection_stats()
    for category, count in stats.items():
        print(f"  {category}: {count} documents")
    print(f"\n  Total: {total_docs} documents ingested")
    print("═══════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(ingest_all())
