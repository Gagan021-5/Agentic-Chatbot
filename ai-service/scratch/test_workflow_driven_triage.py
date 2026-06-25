import sys
import os
import asyncio
import json
from dotenv import load_dotenv

# Load env vars from parent directory .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from services.llm import LLMService
from services.step_router import route

class MockSessionManager:
    def __init__(self):
        self.sessions = {}

    async def get_session(self, session_id):
        return self.sessions.get(session_id)

    async def save_session(self, session):
        self.sessions[session.get("sessionId")] = session

    async def get_or_create_session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "sessionId": session_id,
                "step": 0,
                "history": [],
                "appType": None,
                "extraction": {},
                "deepAnswers": {},
                "verificationMetadata": {}
            }
        return self.sessions[session_id]

    async def reset_app_specific_context(self, session_id):
        session = self.sessions.get(session_id) or {"sessionId": session_id}
        session["step"] = 0
        session["deepAnswers"] = {}
        session["extraction"] = {}
        session["appType"] = None
        self.sessions[session_id] = session
        return session

class MockAppState:
    def __init__(self, llm):
        self.llm = llm
        self.session = MockSessionManager()

async def test_astrology_workflow(app_state):
    print("\n==============================================")
    print("TESTING DOMAIN WORKFLOW: Astrology/Horoscope")
    print("==============================================")
    
    session_id = "test-astrology-workflow"
    
    # Turn 1
    msg_1 = "I want a personal astrology chart calculator and horoscope prediction app"
    session = await app_state.session.get_or_create_session(session_id)
    session["history"].append({"role": "user", "content": msg_1})
    res_1 = await route(session, msg_1, app_state)
    print("Triage rounds:", session.get("triageRounds"))
    print("Current slot key:", session.get("lastSlotKey"))
    print("Reply 1:", res_1.get("reply"))
    print("Dynamic slots list:")
    print(json.dumps(session.get("dynamicSlots"), indent=2))
    
    # Assert astrology workflow slots were identified (support dynamic LLM naming variations)
    slots = [s["key"] for s in session.get("dynamicSlots", [])]
    assert any(k in slots for k in ("prediction_scope", "scope", "prediction_type", "astrology_type")), "Should have astrology prediction fields"
    
    # Turn 2: Answer prediction_scope
    msg_2 = "daily predictions"
    session = await app_state.session.get_or_create_session(session_id)
    session["history"].append({"role": "user", "content": msg_2})
    res_2 = await route(session, msg_2, app_state)
    print("\nTriage rounds:", session.get("triageRounds"))
    print("Current slot key:", session.get("lastSlotKey"))
    print("Reply 2:", res_2.get("reply"))
    print("Captured Answers:", json.dumps(session.get("deepAnswers"), indent=2))
    
    # Verify answers stored
    answers = session.get("deepAnswers", {})
    assert answers.get("prediction_scope") == "daily predictions" or answers.get("prediction_type") == "daily predictions", "Should save the daily prediction answer under correct slot key"

async def test_blog_rewriter_workflow(app_state):
    print("\n==============================================")
    print("TESTING DOMAIN WORKFLOW: Blog Rewriter")
    print("==============================================")
    
    session_id = "test-blog-rewriter-workflow"
    
    # Turn 1
    msg_1 = "I want an app that rewrites tech blogs for kids"
    session = await app_state.session.get_or_create_session(session_id)
    session["history"].append({"role": "user", "content": msg_1})
    res_1 = await route(session, msg_1, app_state)
    print("Triage rounds:", session.get("triageRounds"))
    print("Current slot key:", session.get("lastSlotKey"))
    print("Reply 1:", res_1.get("reply"))
    print("Dynamic slots list:")
    print(json.dumps(session.get("dynamicSlots"), indent=2))
    
    # Assert blog_rewriter workflow slots were identified
    slots = [s["key"] for s in session.get("dynamicSlots", [])]
    assert any(k in slots for k in ("reading_level", "rewrite_goal", "input_source")), "Should have blog rewriter required fields"

async def test_novel_domain_fallback(app_state):
    print("\n==============================================")
    print("TESTING DYNAMIC WORKFLOW: Novel Domain Fallback")
    print("==============================================")
    
    session_id = "test-novel-domain-fallback"
    
    # Turn 1: Novel concept (e.g. analyzes cricket scorecards and generates exciting commentary)
    msg_1 = "I want a tool that takes photos of cricket scorecards and generates exciting commentary"
    session = await app_state.session.get_or_create_session(session_id)
    session["history"].append({"role": "user", "content": msg_1})
    res_1 = await route(session, msg_1, app_state)
    print("Triage rounds:", session.get("triageRounds"))
    print("Current slot key:", session.get("lastSlotKey"))
    print("Reply 1:", res_1.get("reply"))
    print("Dynamic slots list:")
    print(json.dumps(session.get("dynamicSlots"), indent=2))
    
    # Verify that a workflow has been dynamically generated and stored
    assert session.get("dynamicWorkflow") is not None, "A dynamic workflow dictionary must be generated"
    assert len(session.get("dynamicSlots", [])) >= 2, "At least 2 required fields must be generated for dynamic triage"
    
    # Verify that the first question corresponds to the first field of the dynamic slots
    first_slot = session.get("dynamicSlots")[0]["key"]
    assert session.get("lastSlotKey") == first_slot, f"Triage must ask about the first slot '{first_slot}' first"

async def main():
    llm = LLMService()
    app_state = MockAppState(llm)
    
    try:
        await test_astrology_workflow(app_state)
        await test_blog_rewriter_workflow(app_state)
        await test_novel_domain_fallback(app_state)
        print("\n==============================================")
        print("ALL WORKFLOW TRIAGE TESTS PASSED SUCCESSFULLY!")
        print("==============================================")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await llm.close()

if __name__ == "__main__":
    asyncio.run(main())
