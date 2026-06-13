"""
═══════════════════════════════════════════════════════════════
Document Chunker — Splits documents for RAG ingestion
═══════════════════════════════════════════════════════════════
"""

import re
import structlog

logger = structlog.get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separator: str = "\n\n",
) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Strategy:
    1. Split on double newlines (paragraph boundaries) first
    2. If a paragraph is too long, split on single newlines
    3. If still too long, split on sentence boundaries
    4. Apply overlap between adjacent chunks
    """
    if not text or not text.strip():
        return []

    # Step 1: Split on paragraph boundaries
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph exceeds chunk_size, finalize current chunk
        if current_chunk and (len(current_chunk) + len(para) + 2) > chunk_size:
            chunks.append(current_chunk.strip())
            # Overlap: keep last N characters of previous chunk
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
            else:
                current_chunk = para
        elif len(para) > chunk_size:
            # Paragraph itself is too long — split on sentences
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sentence_chunk = ""
            for sentence in sentences:
                if (len(sentence_chunk) + len(sentence) + 1) > chunk_size:
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                else:
                    sentence_chunk = f"{sentence_chunk} {sentence}".strip()
            if sentence_chunk:
                current_chunk = sentence_chunk
        else:
            current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Filter out very short chunks
    chunks = [c for c in chunks if len(c) > 30]

    logger.debug("chunked_text", input_len=len(text), num_chunks=len(chunks))
    return chunks


def chunk_markdown(text: str, chunk_size: int = 512) -> list[dict]:
    """Split markdown by headers, preserving section context.

    Returns list of {content, header, level} dicts.
    """
    if not text or not text.strip():
        return []

    # Split on markdown headers
    sections = re.split(r"^(#{1,4}\s+.+)$", text, flags=re.MULTILINE)

    results = []
    current_header = ""
    current_level = 0
    current_content = ""

    for section in sections:
        header_match = re.match(r"^(#{1,4})\s+(.+)$", section.strip())
        if header_match:
            # Save previous section
            if current_content.strip():
                sub_chunks = chunk_text(current_content.strip(), chunk_size)
                for chunk in sub_chunks:
                    results.append({
                        "content": f"{current_header}\n\n{chunk}".strip() if current_header else chunk,
                        "header": current_header,
                        "level": current_level,
                    })
            current_level = len(header_match.group(1))
            current_header = header_match.group(2).strip()
            current_content = ""
        else:
            current_content += section

    # Don't forget the last section
    if current_content.strip():
        sub_chunks = chunk_text(current_content.strip(), chunk_size)
        for chunk in sub_chunks:
            results.append({
                "content": f"{current_header}\n\n{chunk}".strip() if current_header else chunk,
                "header": current_header,
                "level": current_level,
            })

    return results
