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

# Reconfigure stdout to use UTF-8 on Windows to avoid encoding errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_settings
from rag.vector_store import VectorStoreManager
from rag.chunker import chunk_markdown

settings = get_settings()

# Category → path mapping (can be directory or specific file)
KNOWLEDGE_PATHS = {
    "models": ["knowledge/models", "knowledge/models.md"],
    "prompting": ["knowledge/prompting", "knowledge/prompting.md"],
    "examples": ["knowledge/examples", "knowledge/published.md", "knowledge/marketplace_gold_standards.md"],
    "seo": ["knowledge/seo", "knowledge/seo.md"],
    "marketplace": ["knowledge/marketplace", "knowledge/marketplace_gold_standards.md"],
    "blueprints": ["rag/blueprints"],
}


async def ingest_file(vs: VectorStoreManager, category: str, filepath: str, filename: str) -> int:
    """Ingest a single markdown file into the vector store and return number of chunks."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Chunk the document
    chunks = chunk_markdown(content, chunk_size=512)

    if not chunks:
        print(f"     ⏭️  {filename}: no chunks extracted")
        return 0

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
    print(f"     ✅ {filename}: {len(documents)} chunks ingested")
    return len(documents)


async def ingest_all():
    """Ingest all knowledge base documents into ChromaDB."""
    print("═══════════════════════════════════════════")
    print("  RentPrompts Knowledge Base Ingestion")
    print("═══════════════════════════════════════════")

    # Initialize vector store
    vs = VectorStoreManager(settings)
    await vs.initialize()

    total_docs = 0
    seen_files = set()  # Prevent double indexing the same file under same category

    for category, paths in KNOWLEDGE_PATHS.items():
        print(f"\n  📂 Processing: {category}/")
        
        for rel_path in paths:
            full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel_path)

            if not os.path.exists(full_path):
                continue

            if os.path.isdir(full_path):
                for filename in os.listdir(full_path):
                    if not filename.endswith(".md"):
                        continue

                    filepath = os.path.join(full_path, filename)
                    file_key = (category, filepath)
                    if file_key in seen_files:
                        continue
                    seen_files.add(file_key)

                    try:
                        chunks_count = await ingest_file(vs, category, filepath, filename)
                        total_docs += chunks_count
                    except Exception as e:
                        import traceback
                        print(f"\n❌ Error ingesting {filename} in {category}: {e}")
                        traceback.print_exc()
                        raise e
            elif os.path.isfile(full_path):
                filename = os.path.basename(full_path)
                file_key = (category, full_path)
                if file_key in seen_files:
                    continue
                seen_files.add(file_key)

                try:
                    chunks_count = await ingest_file(vs, category, full_path, filename)
                    total_docs += chunks_count
                except Exception as e:
                    import traceback
                    print(f"\n❌ Error ingesting {filename} in {category}: {e}")
                    traceback.print_exc()
                    raise e

    # Print stats
    print("\n═══════════════════════════════════════════")
    stats = vs.get_collection_stats()
    for category, count in stats.items():
        print(f"  {category}: {count} documents")
    print(f"\n  Total: {total_docs} documents ingested")
    print("═══════════════════════════════════════════")


if __name__ == "__main__":
    asyncio.run(ingest_all())
