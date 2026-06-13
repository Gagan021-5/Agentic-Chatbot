"""
═══════════════════════════════════════════════════════════════
LangGraph Workflow — RentPrompts AI Pipeline StateGraph
═══════════════════════════════════════════════════════════════

Multi-turn conversational agent pipeline for configuring,
optimizing, and registering Rapps.
"""

from __future__ import annotations
import re
import json
import time
from typing import TypedDict, Annotated, Any, Literal, Optional
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from loguru import logger

from data.models import MODELS


# ─── State Definition ──────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Full state flowing through the LangGraph pipeline."""
    session_id: str
    message: str
    history: Annotated[list, add_messages]

    # Active Application Scope Attributes
    app_type: Optional[str]
    app_purpose: Optional[str]
    extraction: dict[str, Any]
    deep_answers: dict[str, Any]

    # Conversation progress tracking
    current_step: int
    recommended_action: str
    reply: str

    # Retrieval and models
    model_id: Optional[str]
    model_guidance: Optional[str]
    rag_documents: list[dict[str, Any]]
    similar_apps: list[dict[str, Any]]
    merged_context: str
    web_research_results: dict[str, Any]

    # Prompt and manifest results
    enhanced_system_prompt: Optional[str]
    enhanced_user_prompt: Optional[str]
    rag_context_injected: Optional[str]
    optimization_notes: list[str]
    extracted_variables: list[dict[str, Any]]

    # Exit flags / completion status
    requirements_complete: bool
    preview_approved: bool
    cms_registered: bool
    processing_time_ms: float


# ─── Node Implementations ──────────────────────────────────

async def intent_classifier_node(state: PipelineState, config: dict) -> dict:
    """Classifies user intent and determines the next execution node."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    msg_clean = message.strip().lower()

    # 1. Quick regex intercepts for fast path
    if msg_clean in ("hi", "hello", "hey", "hola"):
        return {"recommended_action": "HANDLE_GREETING"}
    if msg_clean in ("approve", "publish", "approve app", "looks good", "confirm"):
        return {"recommended_action": "APPROVE"}

    # Check model catalog IDs
    all_model_ids = []
    for cat_models in MODELS.values():
        for m in cat_models:
            all_model_ids.append(m["id"])

    for model_id in all_model_ids:
        if model_id in msg_clean:
            return {"recommended_action": "MODEL_SELECT", "model_id": model_id}

    # 2. Informational questions / off-topic heuristics
    question_prefixes = (
        "how does", "how do", "how is", "what is", "what are", "whats", "what's",
        "why does", "why do", "why is", "tell me about", "explain how", "explain what",
        "can you explain", "how can i", "how do i", "how to"
    )
    is_informational_question = msg_clean.startswith(question_prefixes) or (
        ("?" in msg_clean or msg_clean.startswith(("what", "how", "why", "explain")))
        and not any(phrase in msg_clean for phrase in (
            "i want to build", "i want to create", "i want to make", "i want to start",
            "let's build", "lets build", "let's create", "lets create", "let's make", "lets make",
            "create a", "create an", "build a", "build an", "make a", "make an"
        ))
    )
    if is_informational_question:
        return {"recommended_action": "HANDLE_OFF_TOPIC"}

    # 3. LLM classification call
    history_slice = []
    for h in state.get("history", [])[-8:]:
        role = "assistant" if h.get("role") == "agent" else "user"
        content = h.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        history_slice.append({"role": role, "content": content[:400]})

    system_prompt = """You are an intent classifier for a conversational AI App Builder.
Classify the user's message into one of these actions:
- "HANDLE_OFF_TOPIC": General question about AI, technology, programming, or general chit-chat.
- "HANDLE_GREETING": Hello, hi, greetings.
- "MODEL_SELECT": Selecting or switching an AI model (e.g. "let's use gpt-4o", "change to flux").
- "APPROVE": Approving the prompt preview or asking to publish.
- "BUILD": Providing information about the app they want to build (describing goals, type, inputs, audience).

Return JSON only:
{
  "action": "HANDLE_OFF_TOPIC|HANDLE_GREETING|MODEL_SELECT|APPROVE|BUILD",
  "model_id": "extracted model id if model select, else null"
}"""
    try:
        res = await app_state.llm.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                *history_slice,
                {"role": "user", "content": message}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)
        action = parsed.get("action") or "BUILD"
        m_id = parsed.get("model_id")
        return {
            "recommended_action": action,
            "model_id": m_id if m_id else state.get("model_id")
        }
    except Exception as e:
        logger.warning(f"LLM intent classification failed, falling back: {e}")
        return {"recommended_action": "BUILD"}


async def off_topic_responder_node(state: PipelineState, config: dict) -> PipelineState:
    """Answers informational queries using the RAG knowledge base."""
    app_state = config["configurable"]["app_state"]
    query = state.get("message", "")

    # 1. RAG retrieval
    rag_docs = []
    try:
        rag_results = await app_state.vector_store.search(
            query=query,
            categories=["models", "prompting", "marketplace", "examples", "seo"],
            top_k=3,
        )
        rag_docs = rag_results
    except Exception as e:
        logger.warning(f"RAG search failed in off-topic responder: {e}")

    context = "\n\n".join([d["content"] for d in rag_docs])

    # 2. LLM response
    system_prompt = f"""You are the RentPrompts Help Desk Assistant.
Answer the user's question friendly and naturally. Use the following retrieved knowledge base context to provide accurate answers about the platform, models, or prompt engineering where relevant:
{context}

Keep your answer relatively concise (1-3 paragraphs) and helpful.
Do not prompt the user about their app setup in this message unless relevant."""

    history_slice = []
    for h in state.get("history", [])[-8:]:
        role = "assistant" if h.get("role") == "agent" else "user"
        content = h.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        history_slice.append({"role": role, "content": content[:400]})

    try:
        res = await app_state.llm.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                *history_slice,
                {"role": "user", "content": query}
            ],
            model="llama-3.1-8b-instant"
        )
        reply = res.get("choices", [{}])[0].get("message", {}).get("content", "I am here to help you learn about AI models and our platform.")
    except Exception as e:
        reply = "I'm sorry, I'm having trouble answering your question right now. How can I help you build your app?"

    state["reply"] = reply
    state["rag_documents"] = rag_docs
    return state


async def greeting_node(state: PipelineState) -> PipelineState:
    """Welcomes the user and asks for app specifications."""
    state["reply"] = (
        "Hello! 👋 I'm your App Creator Assistant. I will guide you through designing your application.\n\n"
        "To get started, **what type of AI app would you like to build today?**"
    )
    return state


async def ideation_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 1: Scopes the app purpose and type using extraction heuristics."""
    app_state = config["configurable"]["app_state"]
    message = state.get("message", "")
    history = state.get("history", [])

    app_type = state.get("app_type")
    app_purpose = state.get("app_purpose")
    extraction = state.get("extraction") or {}

    from services.extraction import extract_requirements
    ext = await extract_requirements(app_state.llm, message, history)

    if ext.get("appType") and ext["appType"] != "null":
        app_type = ext["appType"]
    if ext.get("appPurpose") and len(ext["appPurpose"]) > 5:
        app_purpose = ext["appPurpose"]

    extraction.update({k: v for k, v in ext.items() if v is not None and v != "null"})

    state["app_type"] = app_type
    state["app_purpose"] = app_purpose
    state["extraction"] = extraction

    has_purpose = bool(app_purpose and len(app_purpose) > 8)
    has_type = bool(app_type and app_type in ("text", "image", "audio", "video", "vision"))

    if has_purpose and has_type:
        state["requirements_complete"] = True
        state["current_step"] = 2
    else:
        state["requirements_complete"] = False
        state["current_step"] = 1
        if not has_type:
            state["reply"] = (
                f"I've got your goal: \"{app_purpose}\" if that's right. "
                "But I need to know what **type of output** your AI app should generate.\n\n"
                "Please choose one of: Text, Image, Audio, Video, or Vision."
            )
        else:
            state["reply"] = ext.get("suggestedReply") or "Could you describe what your AI app should do or generate in more detail?"

    return state


async def model_selection_node(state: PipelineState) -> PipelineState:
    """Step 2: Renders available models and requests selection."""
    app_type = state.get("app_type", "text")
    model_id = state.get("model_id")

    candidates = MODELS.get(app_type, [])

    if not model_id:
        card_texts = []
        for m in candidates:
            card_texts.append(f"- **{m['name']}** (Cost: {m['cost']} coins) - *{m['desc']}*")

        options = "\n".join(card_texts)
        state["reply"] = (
            f"Requirements verified! For **{app_type}** apps, here are our recommended models:\n\n"
            f"{options}\n\n"
            "Please select a model by typing its name or choosing one to proceed."
        )
    else:
        state["current_step"] = 3

    return state


async def rag_and_optimization_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 3 & 4: Retrieval and Prompt Optimization with variable mapping."""
    app_state = config["configurable"]["app_state"]
    app_purpose = state.get("app_purpose", "")
    app_type = state.get("app_type", "text")
    model_id = state.get("model_id")

    query = f"{app_type} app: {app_purpose}"
    all_docs = []

    # 1. Retrieval
    try:
        categories = ["models", "prompting", "marketplace", "examples", "seo"]
        rag_results = await app_state.vector_store.search(
            query=query,
            categories=categories,
            top_k=5,
        )
        all_docs.extend(rag_results)
    except Exception as e:
        logger.warning(f"RAG search error in optimization: {e}")

    state["rag_documents"] = all_docs

    context_parts = []
    for doc in all_docs:
        context_parts.append(f"[{doc['category'].upper()}] Source: {doc['source']}\n{doc['content']}")
    rag_context = "\n---\n".join(context_parts)
    state["rag_context_injected"] = rag_context

    # 2. Guidance hints
    model_guidance = _get_model_hints(model_id, app_type)
    state["model_guidance"] = model_guidance

    # 3. Optimize Prompt Templates
    system_prompt = f"""You are the RentPrompts Prompt Engineer.
Your task is to craft an optimized, high-fidelity system prompt and user prompt blueprint for a custom AI app.
The app type is {app_type} and its purpose is: {app_purpose}.
The target AI engine is {model_id}.

Here is the retrieved knowledge context:
{rag_context}

Here is the model-specific guidance:
{model_guidance}

You MUST define user inputs/variables in the prompt template using square brackets: [variable_name]
Example: "Write a [tone] birthday message for [recipient_name]."

Return strict JSON only:
{{
  "system_prompt": "enhanced system prompt here",
  "user_prompt": "enhanced user prompt here",
  "optimization_notes": ["note 1", "note 2"]
}}"""

    try:
        res = await app_state.llm.groq_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the optimized prompts."}
            ],
            model="llama-3.1-8b-instant",
            response_format={"type": "json_object"}
        )
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        parsed = json.loads(content)

        state["enhanced_system_prompt"] = parsed.get("system_prompt")
        state["enhanced_user_prompt"] = parsed.get("user_prompt")
        state["optimization_notes"] = parsed.get("optimization_notes") or []
    except Exception as e:
        logger.error(f"Failed to generate optimized prompts: {e}")
        state["enhanced_system_prompt"] = f"You are a helpful {app_purpose} AI assistant."
        state["enhanced_user_prompt"] = f"Generate {app_purpose} output based on [user_input]."
        state["optimization_notes"] = ["Fallback prompt generated due to error."]

    # 4. Variable Extraction via [variable_name]
    combined_prompts = f"{state['enhanced_system_prompt']}\n{state['enhanced_user_prompt']}"
    var_pattern = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*)\]")
    found_vars = list(set(var_pattern.findall(combined_prompts)))

    variables = []
    for var_name in found_vars:
        variables.append({
            "identifier": var_name,
            "display_name": _humanize_variable(var_name),
            "type": _infer_variable_type(var_name),
            "placeholder": _generate_placeholder(var_name, app_type),
            "required": True,
        })
    state["extracted_variables"] = variables
    state["current_step"] = 5

    return state


async def preview_and_registration_node(state: PipelineState, config: dict) -> PipelineState:
    """Step 5 & 6: Previews mock outputs and registers final Rapp via Payload CMS."""
    app_state = config["configurable"]["app_state"]
    action = state.get("recommended_action")
    app_type = state.get("app_type", "text")
    app_purpose = state.get("app_purpose", "")

    if action == "APPROVE" or state.get("preview_approved"):
        payload = {
            "appType": app_type,
            "modelId": state.get("model_id"),
            "costPerRun": 2.0,
            "systemPrompt": state.get("enhanced_system_prompt"),
            "userPrompt": state.get("enhanced_user_prompt"),
            "appName": f"{app_purpose.title()[:30]} Creator",
            "appDescription": f"AI app for {app_purpose}",
            "tags": [app_type, "automated"],
            "publishedAt": datetime.now(timezone.utc).isoformat(),
        }
        try:
            res = await app_state.cms.create_rapp(payload)
            state["cms_registered"] = True
            state["reply"] = (
                f"## 🎉 RAPP Registered Successfully!\n\n"
                f"Your application **\"{payload['appName']}\"** has been registered in the CMS catalog.\n\n"
                f"- **Model:** {state.get('model_id')}\n"
                f"- **Rapp ID:** {res.get('id', 'N/A')}\n"
                f"- **Status:** Live & Ready! 🚀"
            )
        except Exception as e:
            logger.error(f"CMS registration failed: {e}")
            state["reply"] = "Rapp registration failed due to CMS connection error. Please try again."
    else:
        mock_assets = {
            "text": "Generated text output preview goes here...",
            "image": "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=500",
            "audio": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
            "video": "https://www.w3schools.com/html/mov_bbb.mp4",
            "vision": "Analyzed vision labels: [Object, Pattern, Texture]",
        }
        mock_url = mock_assets.get(app_type, mock_assets["text"])

        state["reply"] = (
            f"### 📋 Prompt Preview & Mock Assets\n\n"
            f"**Enhanced System Prompt:**\n```\n{state.get('enhanced_system_prompt')}\n```\n\n"
            f"**Enhanced User Prompt:**\n```\n{state.get('enhanced_user_prompt')}\n```\n\n"
            f"**Mock Output Preview:**\n"
            f"{mock_url}\n\n"
            f"If you like this configuration, type **Approve** or **Publish** to deploy it!"
        )

    return state


# ─── Routing Logic ──────────────────────────────────────────

def route_conditional_edge(state: PipelineState) -> str:
    """Routes initial intent classifier decisions."""
    action = state.get("recommended_action")
    if action == "HANDLE_OFF_TOPIC":
        return "off_topic_responder"
    if action == "HANDLE_GREETING":
        return "greeting"

    step = state.get("current_step", 1)
    if step == 1:
        return "ideation"
    if step == 2:
        return "model_selection"
    if step >= 3 and not state.get("extracted_variables"):
        return "rag_and_optimization"
    return "preview_and_registration"


def should_continue_from_ideation(state: PipelineState) -> Literal["model_selection", "end"]:
    """Gates continuation based on requirement completeness."""
    if state.get("requirements_complete"):
        return "model_selection"
    return "end"


def should_continue_from_model(state: PipelineState) -> Literal["rag_and_optimization", "end"]:
    """Gates model choice requirements."""
    if state.get("model_id"):
        return "rag_and_optimization"
    return "end"


# ─── Graph Builder ──────────────────────────────────────────

def build_pipeline_graph(vector_store=None) -> StateGraph:
    """Build the conversational StateGraph workflow."""
    graph = StateGraph(PipelineState)

    # Register Nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("off_topic_responder", off_topic_responder_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("ideation", ideation_node)
    graph.add_node("model_selection", model_selection_node)
    graph.add_node("rag_and_optimization", rag_and_optimization_node)
    graph.add_node("preview_and_registration", preview_and_registration_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges(
        "intent_classifier",
        route_conditional_edge,
        {
            "off_topic_responder": "off_topic_responder",
            "greeting": "greeting",
            "ideation": "ideation",
            "model_selection": "model_selection",
            "rag_and_optimization": "rag_and_optimization",
            "preview_and_registration": "preview_and_registration",
        }
    )

    graph.add_conditional_edges(
        "ideation",
        should_continue_from_ideation,
        {
            "model_selection": "model_selection",
            "end": END,
        }
    )

    graph.add_conditional_edges(
        "model_selection",
        should_continue_from_model,
        {
            "rag_and_optimization": "rag_and_optimization",
            "end": END,
        }
    )

    graph.add_edge("rag_and_optimization", "preview_and_registration")
    graph.add_edge("preview_and_registration", END)
    graph.add_edge("off_topic_responder", END)
    graph.add_edge("greeting", END)

    return graph.compile()


# ─── Helper Functions ───────────────────────────────────────

def _get_model_hints(model_id: str | None, app_type: str) -> str:
    """Return model-specific prompt engineering guidance."""
    if not model_id:
        return ""

    hints = {
        "flux-schnell": "Flux Schnell is ultra-fast. Keep prompts concise (40-80 words). Focus on style.",
        "flux-2-pro": "Flux 2 Pro supports editing. Use descriptive visual language.",
        "imagen-4": "Imagen 4 excels at photorealism. Use natural language descriptions.",
        "sdxl": "SDXL responds well to weighted prompts. Use specific art styles.",
        "recraft-v4-pro": "Recraft v4 Pro is design-focused with strong text rendering.",
        "gpt-5.2": "GPT-5.2 is coding/agentic flagship. Use structured instructions.",
        "gpt-5.1": "GPT-5.1 has deep reasoning. Ask for step-by-step analysis.",
        "gpt-4o": "GPT-4o is versatile multimodal. Keep prompts task-focused.",
        "llama3.3-70b": "LLaMA 3.3 70B responds well to structured system prompts.",
    }
    return hints.get(model_id, f"Use best practices for {app_type} generation.")


def _humanize_variable(name: str) -> str:
    return name.replace("_", " ").title()


def _infer_variable_type(name: str) -> str:
    name_lower = name.lower()
    if any(k in name_lower for k in ["image", "photo", "picture", "url"]):
        return "image_url"
    if any(k in name_lower for k in ["count", "number", "quantity", "age", "year"]):
        return "number"
    if any(k in name_lower for k in ["enabled", "active", "include"]):
        return "boolean"
    return "string"


def _generate_placeholder(name: str, app_type: str) -> str:
    name_lower = name.lower()
    placeholders = {
        "topic": "e.g., The future of AI in healthcare",
        "style": "e.g., photorealistic, anime, watercolor",
        "tone": "e.g., professional, casual, humorous",
        "language": "e.g., English, Hindi, Spanish",
        "audience": "e.g., college students, business executives",
    }
    for key, placeholder in placeholders.items():
        if key in name_lower:
            return placeholder
    return f"Enter {_humanize_variable(name).lower()}"
