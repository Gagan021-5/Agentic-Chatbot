"""
═══════════════════════════════════════════════════════════════
Pydantic Schemas — Request/Response models for all endpoints
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from typing import Optional, Any, Literal
from pydantic import BaseModel, Field


# ─── Verification Status Tracking ───────────────────────────

VerificationStatus = Literal["missing", "inferred", "explicit"]


class VerificationMetadataSchema(BaseModel):
    """Tracks lineage status of critical architecture keys.
    
    Each field maps to a VerificationStatus indicating whether the value
    was explicitly declared by the user, inferred from context, or is
    still unresolved.
    """
    app_type: VerificationStatus = "missing"
    ingestion_vector: VerificationStatus = "missing"
    budget: VerificationStatus = "missing"


# ─── Common ─────────────────────────────────────────────────

class VariableSchema(BaseModel):
    """A single extracted variable from a prompt template."""
    identifier: str = Field(..., description="Snake_case machine identifier")
    display_name: str = Field(..., description="Human-readable label")
    type: str = Field(default="string", description="Variable type: string, number, boolean, enum, image_url")
    placeholder: str = Field(default="", description="Example value hint")
    required: bool = Field(default=True)
    enum_values: list[str] | None = Field(default=None, description="Possible values for enum type")


class RetrievedDocument(BaseModel):
    """A document chunk retrieved from the knowledge base."""
    content: str
    source: str
    category: str
    relevance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── /chat ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Full LangGraph workflow request."""
    session_id: str
    message: str
    app_type: str | None = None
    app_purpose: str | None = None
    model_id: str | None = None
    extraction: dict[str, Any] | None = None
    deep_answers: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    ingestion_vector: str | None = None
    verification_metadata: VerificationMetadataSchema | None = None


class ChatResponse(BaseModel):
    """LangGraph workflow response."""
    session_id: str
    reply: Optional[str] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    enhanced_context: dict[str, Any] = Field(default_factory=dict)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    optimized_prompt: dict[str, Any] | None = None
    variables: list[VariableSchema] = Field(default_factory=list)
    web_research: dict[str, Any] | None = None
    processing_time_ms: float = 0
    app_type: str | None = None
    app_purpose: str | None = None
    model_id: str | None = None
    extraction: dict[str, Any] | None = None
    deep_answers: dict[str, Any] | None = None
    ingestion_vector: str | None = None
    verification_metadata: VerificationMetadataSchema | None = None


# ─── /retrieve-context ─────────────────────────────────────

class RetrievalRequest(BaseModel):
    """RAG context retrieval request."""
    session_id: str
    query: str
    app_type: str | None = None
    model_id: str | None = None
    categories: list[str] = Field(
        default_factory=lambda: ["models", "prompting", "examples", "seo", "marketplace"],
        description="Knowledge base categories to search"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    boost_gold_standards: bool = Field(default=False, description="Increase priority for marketplace gold standards")


class RetrievalResponse(BaseModel):
    """RAG context retrieval response."""
    documents: list[RetrievedDocument]
    query: str
    total_found: int
    processing_time_ms: float = 0


# ─── /optimize-prompt ──────────────────────────────────────

class PromptOptimizeRequest(BaseModel):
    """Prompt optimization request for RAG-enhanced prompt generation."""
    session_id: str
    app_type: str
    app_purpose: str
    model_id: str | None = None
    variables: list[dict[str, str]] = Field(default_factory=list)
    deep_answers: dict[str, Any] = Field(default_factory=dict)
    existing_system_prompt: str | None = None
    existing_user_prompt: str | None = None
    edit_instruction: str | None = None


class PromptOptimizeResponse(BaseModel):
    """Enhanced prompt data with RAG-injected context."""
    enhanced_system_prompt: str | None = None
    enhanced_user_prompt: str | None = None
    rag_context_injected: str = ""
    model_guidance: str | None = None
    similar_apps: list[dict[str, Any]] = Field(default_factory=list)
    optimization_notes: list[str] = Field(default_factory=list)
    variables: list[VariableSchema] = Field(default_factory=list)
    processing_time_ms: float = 0


# ─── /web-research ─────────────────────────────────────────

class WebResearchRequest(BaseModel):
    """Web research request — agent decides when this is needed."""
    session_id: str
    query: str
    context: str | None = None
    max_results: int = Field(default=5, ge=1, le=10)


class WebResearchResponse(BaseModel):
    """Summarized web research results."""
    query: str
    summary: str
    sources: list[dict[str, str]] = Field(default_factory=list)
    raw_results: list[dict[str, Any]] = Field(default_factory=list)
    cached: bool = False
    processing_time_ms: float = 0


# ─── /health ───────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Service health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    services: dict[str, str] = Field(default_factory=dict)
