import asyncio
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Add the parent directory to the path so we can import services
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.intent_engine import try_fast_path, enforce_prd_rules
from services.step_router import route, _show_models

class MockLLM:
    has_groq = True
    async def groq_completion(self, *args, **kwargs):
        # Fallback to a mock response if needed
        return {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "orchestrate_pipeline",
                            "arguments": '{"recommended_action": "SHOW_MODEL_CARDS", "app_type": "text", "confidence": "HIGH", "reasoning": "Mock prompt response", "edit_scope": "PATCH_PROMPT"}'
                        }
                    }]
                }
            }]
        }

class MockSessionManager:
    def __init__(self):
        self.sessions = {}

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def save_session(self, session):
        self.sessions[session.get("sessionId")] = session

    async def delete_session(self, session_id):
        self.sessions.pop(session_id, None)

    async def reset_app_specific_context(self, session_id):
        session = self.sessions.get(session_id) or {"sessionId": session_id}
        session["step"] = 0
        session["deepAnswers"] = {}
        session["extraction"] = {}
        session["appType"] = None
        self.sessions[session_id] = session
        return session

class MockAppState:
    def __init__(self):
        self.llm = MockLLM()
        self.session = MockSessionManager()

async def run_tests():
    print("=== TEST 1: try_fast_path modality correction ===")
    
    # 1. Simple text app correction
    session1 = {"appType": "image", "formConfirmed": True}
    res1 = try_fast_path("it is a text app", session1)
    print("it is a text app:", res1)
    assert res1 is not None
    assert res1["app_type"] == "text"
    assert res1["recommended_action"] == "SHOW_MODEL_CARDS"
    assert res1["_source"] == "fast_path"

    # 2. Text app correction without formConfirmed
    session2 = {"appType": "image", "formConfirmed": False}
    res2 = try_fast_path("i want a text app", session2)
    print("i want a text app (no formConfirmed):", res2)
    assert res2 is not None
    assert res2["app_type"] == "text"
    assert res2["recommended_action"] == "GATHER_REQUIREMENTS"

    # 3. Negation handling: "not an image app" (should shift to text since it is the alternative in lower or default)
    res3 = try_fast_path("not an image app, make it text", session1)
    print("not an image app, make it text:", res3)
    assert res3 is not None
    assert res3["app_type"] == "text"

    # 4. Negation handling: "text app instead of image"
    res4 = try_fast_path("text app instead of image", session1)
    print("text app instead of image:", res4)
    assert res4 is not None
    assert res4["app_type"] == "text"

    # 5. Chip types matching with formConfirmed = True should trigger SHOW_MODEL_CARDS
    res5 = try_fast_path("text app", session1)
    print("text app (chip types):", res5)
    assert res5 is not None
    assert res5["recommended_action"] == "SHOW_MODEL_CARDS"

    print("\n=== TEST 2: enforce_prd_rules State Preservation Bypass ===")
    
    # Normally, if appType is image and we don't pivot, enforce_prd_rules locks app_type to image
    decision = {
        "recommended_action": "SHOW_MODEL_CARDS",
        "app_type": "text",
        "confidence": "HIGH",
        "_source": "llm" # Not fast path
    }
    enforced = enforce_prd_rules(decision, session1)
    print("PRD rule check (LLM, should keep image):", enforced["app_type"])
    assert enforced["app_type"] == "image"

    # If it is fast_path, it should bypass the preservation override and allow text
    decision_fast = {
        "recommended_action": "SHOW_MODEL_CARDS",
        "app_type": "text",
        "confidence": "HIGH",
        "_source": "fast_path"
    }
    enforced_fast = enforce_prd_rules(decision_fast, session1)
    print("PRD rule check (fast_path, should allow text):", enforced_fast["app_type"])
    assert enforced_fast["app_type"] == "text"

    print("\n=== TEST 3: step_router _show_models bypass ===")
    # History contains "image" word from previous debug runs, but appType is text
    session_history = {
        "appType": "text",
        "history": [
            {"role": "user", "content": "creates fantasy world maps"},
            {"role": "agent", "content": "here is an image"},
            {"role": "user", "content": "it is a text app"} # Most recent has text correction
        ],
        "extraction": {
            "appPurpose": "Daily horoscopes, birth chart interpretations, and planetary transit reports",
            "appType": "text"
        }
    }
    app_state = MockAppState()
    result = await _show_models(session_history, app_state)
    print("Resulting appType in session after show_models:", session_history["appType"])
    assert session_history["appType"] == "text"

    print("\n=== TEST 4: step_router route execution ===")
    session_full = {
        "sessionId": "test-session-123",
        "appType": "image",
        "step": 1,
        "formConfirmed": True,
        "deepAnswers": {
            "budgetPreference": "premium"
        },
        "history": [
            {"role": "user", "content": "generate high-quality AI images"},
            {"role": "agent", "content": "Please select a model"},
        ],
        "extraction": {
            "appPurpose": "fantasy maps",
            "appType": "image",
            "budget": "premium"
        }
    }
    
    # Save session first in our mock db
    await app_state.session.save_session(session_full)
    
    # User corrects format to text app (non-pivot, just format correction)
    final_payload = await route(session_full, "it is a text app", app_state)
    print("Route response appType in session:", session_full["appType"])
    print("Route response reply start:", final_payload.get("reply", "")[:100])
    assert session_full["appType"] == "text"
    assert "Text" in final_payload.get("reply", "")

    print("\n=== TEST 5: step_router drastic pivot execution ===")
    session_pivot = {
        "sessionId": "test-session-456",
        "appType": "image",
        "step": 1,
        "formConfirmed": True,
        "deepAnswers": {
            "budgetPreference": "premium",
            "landmass_style": "archipelago"
        },
        "history": [
            {"role": "user", "content": "generate high-quality AI images"},
            {"role": "agent", "content": "Please select a model"},
        ],
        "extraction": {
            "appPurpose": "fantasy maps", # creative category
            "appType": "image",
            "budget": "premium"
        }
    }
    await app_state.session.save_session(session_pivot)
    
    # User switches category radically to "workout planner" (functional category)
    pivot_payload = await route(session_pivot, "I want an app that creates workout plans", app_state)
    print("Pivot response appType in session:", session_pivot["appType"])
    print("Pivot response reply:", pivot_payload.get("reply", ""))
    
    # Assertions
    assert session_pivot["appType"] == "text"
    assert "pivoted from" in pivot_payload.get("reply", "")
    assert "Creative" in pivot_payload.get("reply", "")
    assert "Functional" in pivot_payload.get("reply", "")
    # Check that previous variables like landmass_style have been flushed
    assert "landmass_style" not in session_pivot.get("deepAnswers", {})

    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    asyncio.run(run_tests())
