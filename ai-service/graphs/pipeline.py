"""
═══════════════════════════════════════════════════════════════
LangGraph Workflow — RentPrompts AI Pipeline StateGraph
═══════════════════════════════════════════════════════════════

START → RequirementAnalysis → ModelSelection → Retrieval
       ├─ InternalRAG
       ├─ HistoricalAppSearch
       └─ WebSearch
       → PromptEngineering → VariableExtraction → Output → END
"""

from __future__ import annotations
import time
import structlog
from typing import TypedDict, Annotated, Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

logger = structlog.get_logger(__name__)


# ─── State Definition ──────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""

    # --- Input ---
    session_id: str
    message: str
    app_type: str | None
    app_purpose: str | None
    model_id: str | None
    extraction: dict[str, Any]
    deep_answers: dict[str, Any]
    history: list[dict[str, Any]]

    # --- Intermediate ---
    requirements_complete: bool
    model_guidance: str
    rag_documents: list[dict[str, Any]]
    web_research_results: dict[str, Any]
    merged_context: str

    # --- Output ---
    enhanced_system_prompt: str | None
    enhanced_user_prompt: str | None
    rag_context_injected: str
    optimization_notes: list[str]
    extracted_variables: list[dict[str, Any]]
    similar_apps: list[dict[str, Any]]
    processing_time_ms: float


# ─── Node Implementations ──────────────────────────────────

async def requirement_analysis_node(state: PipelineState) -> PipelineState:
    """Analyze incoming requirements and determine readiness."""
    logger.info("node_requirement_analysis", session_id=state.get("session_id"))

    app_purpose = state.get("app_purpose", "")
    extraction = state.get("extraction", {})

    # Determine if we have enough to proceed
    has_purpose = bool(app_purpose and len(app_purpose) > 10)
    has_type = bool(state.get("app_type"))
    has_model = bool(state.get("model_id"))

    state["requirements_complete"] = has_purpose and has_type
    state["optimization_notes"] = []

    if not has_purpose:
        state["optimization_notes"].append("App purpose is too vague for optimization")
    if not has_type:
        state["optimization_notes"].append("App type not yet determined")

    return state


async def model_selection_node(state: PipelineState) -> PipelineState:
    """Prepare model-specific guidance for prompt engineering."""
    logger.info("node_model_selection", model_id=state.get("model_id"))

    model_id = state.get("model_id")
    app_type = state.get("app_type", "text")

    # Build model guidance context
    model_hints = _get_model_hints(model_id, app_type)
    state["model_guidance"] = model_hints

    return state


async def retrieval_node(state: PipelineState, vector_store) -> PipelineState:
    """Parallel retrieval from internal RAG, historical apps, and web search."""
    logger.info("node_retrieval", session_id=state.get("session_id"))

    app_purpose = state.get("app_purpose", "")
    app_type = state.get("app_type", "text")
    model_id = state.get("model_id")

    query = f"{app_type} app: {app_purpose}"
    all_docs = []

    # 1. Internal RAG — model docs + prompting guides
    categories_to_search = ["models", "prompting", "marketplace"]
    if model_id:
        categories_to_search.insert(0, "models")

    try:
        rag_results = await vector_store.search(
            query=query,
            categories=categories_to_search,
            top_k=5,
        )
        all_docs.extend(rag_results)
    except Exception as e:
        logger.warning("rag_search_error", error=str(e))

    # 2. Historical app search — similar published apps
    try:
        example_results = await vector_store.search(
            query=query,
            categories=["examples"],
            top_k=3,
        )
        state["similar_apps"] = [
            {"content": r["content"], "source": r["source"], "score": r["relevance_score"]}
            for r in example_results
        ]
        all_docs.extend(example_results)
    except Exception as e:
        logger.warning("examples_search_error", error=str(e))

    # 3. SEO patterns
    try:
        seo_results = await vector_store.search(
            query=f"marketplace listing for {app_type} {app_purpose}",
            categories=["seo"],
            top_k=2,
        )
        all_docs.extend(seo_results)
    except Exception as e:
        logger.warning("seo_search_error", error=str(e))

    state["rag_documents"] = all_docs
    return state


async def prompt_engineering_node(state: PipelineState) -> PipelineState:
    """Enhance prompt with retrieved context and model-specific guidance."""
    logger.info("node_prompt_engineering", session_id=state.get("session_id"))

    rag_docs = state.get("rag_documents", [])
    model_guidance = state.get("model_guidance", "")
    app_purpose = state.get("app_purpose", "")
    app_type = state.get("app_type", "text")
    existing_system = state.get("enhanced_system_prompt") or ""
    existing_user = state.get("enhanced_user_prompt") or ""

    # Build RAG context injection
    context_parts = []

    if model_guidance:
        context_parts.append(f"MODEL GUIDANCE:\n{model_guidance}")

    # Group retrieved docs by category
    by_category: dict[str, list[str]] = {}
    for doc in rag_docs:
        cat = doc.get("category", "general")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(doc["content"])

    for cat, contents in by_category.items():
        label = cat.upper().replace("_", " ")
        combined = "\n---\n".join(contents[:3])  # Max 3 per category
        context_parts.append(f"{label} CONTEXT:\n{combined}")

    rag_context = "\n\n".join(context_parts) if context_parts else ""
    state["rag_context_injected"] = rag_context

    # Build optimization notes
    notes = state.get("optimization_notes", [])
    if rag_docs:
        notes.append(f"Retrieved {len(rag_docs)} relevant documents from knowledge base")
    if model_guidance:
        notes.append(f"Applied model-specific guidance for {state.get('model_id', 'unknown')}")

    state["optimization_notes"] = notes
    return state


async def variable_extraction_node(state: PipelineState) -> PipelineState:
    """Extract and normalize variables from prompt templates."""
    logger.info("node_variable_extraction", session_id=state.get("session_id"))

    import re

    # Extract $$variables from existing prompts
    system_prompt = state.get("enhanced_system_prompt") or ""
    user_prompt = state.get("enhanced_user_prompt") or ""
    combined = f"{system_prompt}\n{user_prompt}"

    # Find all $$variable patterns
    var_pattern = re.compile(r"\$\$([a-zA-Z_][a-zA-Z0-9_]*)")
    found_vars = list(set(var_pattern.findall(combined)))

    variables = []
    for var_name in found_vars:
        variables.append({
            "identifier": var_name,
            "display_name": _humanize_variable(var_name),
            "type": _infer_variable_type(var_name),
            "placeholder": _generate_placeholder(var_name, state.get("app_type", "text")),
            "required": True,
        })

    state["extracted_variables"] = variables
    return state


async def output_node(state: PipelineState) -> PipelineState:
    """Finalize output and compute metrics."""
    logger.info("node_output", session_id=state.get("session_id"))
    # Output is already assembled in state
    return state


# ─── Routing Logic ──────────────────────────────────────────

def should_retrieve(state: PipelineState) -> Literal["retrieve", "skip_retrieval"]:
    """Decide if retrieval is needed based on requirements."""
    if state.get("requirements_complete"):
        return "retrieve"
    return "skip_retrieval"


# ─── Graph Builder ──────────────────────────────────────────

def build_pipeline_graph(vector_store) -> StateGraph:
    """Build the LangGraph StateGraph for the RentPrompts AI pipeline."""

    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("requirement_analysis", requirement_analysis_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("retrieval", lambda s: retrieval_node(s, vector_store))
    graph.add_node("prompt_engineering", prompt_engineering_node)
    graph.add_node("variable_extraction", variable_extraction_node)
    graph.add_node("output", output_node)

    # Define edges
    graph.set_entry_point("requirement_analysis")

    graph.add_edge("requirement_analysis", "model_selection")

    graph.add_conditional_edges(
        "model_selection",
        should_retrieve,
        {
            "retrieve": "retrieval",
            "skip_retrieval": "prompt_engineering",
        },
    )

    graph.add_edge("retrieval", "prompt_engineering")
    graph.add_edge("prompt_engineering", "variable_extraction")
    graph.add_edge("variable_extraction", "output")
    graph.add_edge("output", END)

    return graph.compile()


# ─── Helper Functions ───────────────────────────────────────

def _get_model_hints(model_id: str | None, app_type: str) -> str:
    """Return model-specific prompt engineering guidance."""
    if not model_id:
        return ""

    # Model-specific prompt hints based on the actual RentPrompts model catalog
    hints = {
        # Image models
        "flux-schnell": "Flux Schnell is ultra-fast. Keep prompts concise (40-80 words). Focus on subject + style + lighting. Avoid verbose instructions.",
        "flux-2-pro": "Flux 2 Pro supports editing. Use descriptive visual language. Include camera angle, lighting direction, and material textures.",
        "imagen-4": "Imagen 4 excels at photorealism. Use natural language descriptions. Specify real-world references. Include 'photorealistic, natural lighting'.",
        "sdxl": "SDXL responds well to weighted prompts. Use specific art styles. Include resolution keywords: '4K, ultra-detailed, sharp focus'.",
        "recraft-v4-pro": "Recraft v4 Pro is design-focused with strong text rendering. Great for logos and branded content.",
        "vgpt-image-2": "vGPT Image 2 supports inpainting/editing. Structure prompts with clear subject, action, and environment.",
        "kling-image-v3": "Kling Image v3 excels at creative, detailed compositions. Use vivid descriptive language.",
        "nano-banana-2": "Nano Banana 2 by Google. Fast generation. Works well with structured scene descriptions.",
        "seedream-5-lite": "Seedream 5 Lite supports multi-image generation with reasoning. Include design intent.",

        # Text models
        "gpt-5.2": "GPT-5.2 is the coding/agentic flagship. Use structured instructions with clear output format requirements.",
        "gpt-5.1": "GPT-5.1 has deep reasoning. Best for complex analysis, long-context tasks. Be explicit about reasoning steps.",
        "gpt-4o": "GPT-4o is versatile multimodal. Works across text, vision, and chat. Keep prompts clear and task-focused.",
        "gpt-4o-mini": "GPT-4o Mini is compact. Keep prompts focused and concise. Good for simple Q&A and content generation.",
        "gpt-4.1-nano": "GPT-4.1 Nano is ultra-fast. Short prompts work best. Ideal for simple text generation tasks.",
        "llama3.3-70b": "LLaMA 3.3 70B is open-source. Responds well to structured system prompts. Explicit output formatting needed.",
        "kimi-k2-thinking": "Kimi K2 Thinking specializes in deep reasoning. Use chain-of-thought prompts. Ask for step-by-step analysis.",
        "grok-4": "Grok 4 is xAI's advanced model. Large context window. Good for complex multi-step tasks.",
        "minimax-m2.7": "MiniMax M2.7 is fast and multimodal. Keep prompts efficient. Good for real-time chat applications.",

        # Audio models
        "orpheus-tts": "Orpheus TTS is emotionally expressive. Include tone markers in the script: [excited], [solemn], [warm].",
        "kokoro-82m": "Kokoro 82M is efficient multilingual TTS. Specify language and speaking pace in the prompt.",
        "lyria-3-pro": "Lyria 3 Pro is for music generation. Describe genre, tempo, mood, instruments, and duration.",
        "tts-1.5-max": "TTS 1.5 Max produces human-like speech. Use natural conversational scripts without markdown formatting.",
        "stable-audio": "Stable Audio generates from text prompts. Describe the audio scene: environment, instruments, mood, duration.",

        # Video models
        "veo3": "Veo 3 is Google's most advanced video AI. Describe scenes cinematically. Include camera movements and transitions.",
        "veo-3-fast": "Veo 3 Fast is optimized for speed. Keep scene descriptions focused. One clear visual per prompt.",
        "seedance-2.0": "Seedance 2.0 supports image-to-video. Describe the motion and transformation clearly.",
        "gen-4.5": "Gen 4.5 by Runway. Cinematic quality. Use film terminology: 'dolly shot', 'rack focus', 'fade to black'.",
        "wan-2.2-fast": "Wan 2.2 Fast is the cheapest video option. Simple scene descriptions work best.",

        # Vision models
        "gpt-4.1-vision": "GPT-4.1 Vision excels at image analysis. Prompt should specify what to extract or analyze in the image.",
    }

    return hints.get(model_id, f"Use best practices for {app_type} generation with {model_id}.")


def _humanize_variable(name: str) -> str:
    """Convert snake_case to human-readable title case."""
    return name.replace("_", " ").title()


def _infer_variable_type(name: str) -> str:
    """Infer variable type from its name."""
    name_lower = name.lower()
    if any(k in name_lower for k in ["image", "photo", "picture", "url"]):
        return "image_url"
    if any(k in name_lower for k in ["count", "number", "quantity", "age", "year"]):
        return "number"
    if any(k in name_lower for k in ["enabled", "active", "include"]):
        return "boolean"
    return "string"


def _generate_placeholder(name: str, app_type: str) -> str:
    """Generate a helpful placeholder for a variable."""
    name_lower = name.lower()
    placeholders = {
        "topic": "e.g., The future of AI in healthcare",
        "style": "e.g., photorealistic, anime, watercolor",
        "tone": "e.g., professional, casual, humorous",
        "language": "e.g., English, Hindi, Spanish",
        "audience": "e.g., college students, business executives",
        "format": "e.g., bullet points, essay, script",
        "subject": "e.g., A majestic lion on a cliff",
        "color_scheme": "e.g., warm earth tones, vibrant neon",
        "background": "e.g., sunset over mountains, studio lighting",
        "company_name": "e.g., Acme Inc.",
        "industry": "e.g., Technology, Healthcare, Education",
    }

    for key, placeholder in placeholders.items():
        if key in name_lower:
            return placeholder

    return f"Enter {_humanize_variable(name).lower()}"
