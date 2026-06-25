"""Helpers for deciding when an uploaded artifact is required for an app's workflow."""
from __future__ import annotations

from typing import Optional


def is_creation_workflow(app_purpose: Optional[str]) -> bool:
    if not app_purpose:
        return False
    p = app_purpose.lower()
    creation_keywords = (
        "logo", "thumbnail", "poster", "illustration", "avatar", "icon",
        "mockup", "design", "generate image", "generate", "create", "make",
        "render", "brand logo", "brand identity", "graphic", "banner",
        "audiobook", "blog", "write", "writer", "story", "script", "planner",
    )
    analysis_keywords = (
        "review", "analy", "analysis", "audit", "evaluate", "ocr", "inspect", "score",
        "resume", "cv", "pitch deck", "invoice", "receipt", "menu", "document",
    )
    if any(k in p for k in creation_keywords) and not any(a in p for a in analysis_keywords):
        return True
    return False


def requires_input_artifact(app_type: Optional[str], app_purpose: Optional[str]) -> bool:
    """Return True only when the uploaded artifact is the actual subject of analysis/review.

    Rules:
    - If the purpose contains explicit analysis/review keywords (resume review, pitch deck, OCR, invoice, receipt, menu, document analysis), require artifact.
    - If the app_type is explicitly 'vision', require artifact.
    - Otherwise, generation workflows (logo generator, blog writer, audiobook, thumbnail generator, etc.) do NOT require artifact.
    """
    atype = (str(app_type or "").lower() or "").strip()
    purpose = (str(app_purpose or "").lower() or "").strip()

    # Analysis / review keywords that imply the uploaded file is the subject
    true_keywords = (
        "review", "audit", "analy", "ocr", "scanner", "detect", "evaluate",
        "evaluation", "score", "grade", "critique", "diagnose", "resume", "cv",
        "pitch deck", "deck", "invoice", "receipt", "menu", "document", "forms?",
    )
    for kw in true_keywords:
        if kw in purpose:
            return True

    # If the orchestrator explicitly classified the app as vision analysis, require artifact
    if atype == "vision":
        return True

    # Otherwise, do not require artifact for common generator workflows
    return False
