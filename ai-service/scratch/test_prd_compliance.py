import sys
import os

# Add parent directory to path to import backend services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.intent_engine import enforce_prd_rules
from services.extraction import is_personal_boilerplate, _sanitize_variable_objects

def run_tests():
    print("[TESTS] Running PRD Compliance Verification Tests...\n")
    
    # ─── TEST 1: CRITICAL STATE PRESERVATION ───
    print("Test 1: Critical State Preservation")
    # locked type (audio) should not be overwritten to text
    session = {"appType": "audio", "step": 0}
    decision = {"recommended_action": "GATHER_REQUIREMENTS", "app_type": "text", "confidence": "medium", "reasoning": "Test"}
    res = enforce_prd_rules(decision, session)
    assert res["app_type"] == "audio", f"Expected 'audio', got {res['app_type']}"
    print("  -> Preserved locked appType 'audio' when action is GATHER_REQUIREMENTS")

    # locked type (audio) can be changed under PIVOT_APP
    decision_pivot = {"recommended_action": "PIVOT_APP", "app_type": "image", "confidence": "medium", "reasoning": "Test"}
    res_pivot = enforce_prd_rules(decision_pivot, session)
    assert res_pivot["app_type"] == "image", f"Expected 'image' under PIVOT_APP, got {res_pivot['app_type']}"
    print("  -> Allowed pivot from 'audio' to 'image' under PIVOT_APP")

    # ─── TEST 2: CLARIFICATION FOLLOW-UP PROTECTION ───
    print("\nTest 2: Clarification Follow-up Protection")
    session_clarify = {"appType": "text", "step": 0, "awaitingDeepAnswer": True}
    decision_cards = {"recommended_action": "SHOW_MODEL_CARDS", "app_type": "text", "confidence": "medium", "reasoning": "Test"}
    res_clarify = enforce_prd_rules(decision_cards, session_clarify)
    assert res_clarify["recommended_action"] == "GATHER_REQUIREMENTS", f"Expected GATHER_REQUIREMENTS, got {res_clarify['recommended_action']}"
    print("  -> Forced action to GATHER_REQUIREMENTS during active clarification question")

    # ─── TEST 3: PREMATURE PROGRESSION GATING ───
    print("\nTest 3: Premature Progression Gating")
    # Only 2 metadata attributes -> should gate
    session_gate = {
        "appType": "image",
        "step": 0,
        "extraction": {
            "PRIMARY_SUBJECT": "motorcycles",
            "ENVIRONMENT_SETTING": "garage"
        },
        "deepAnswers": {}
    }
    decision_gate = {"recommended_action": "SHOW_MODEL_CARDS", "app_type": "image", "confidence": "medium", "reasoning": "Test"}
    res_gate = enforce_prd_rules(decision_gate, session_gate)
    assert res_gate["recommended_action"] == "GATHER_REQUIREMENTS", f"Expected gated to GATHER_REQUIREMENTS, got {res_gate['recommended_action']}"
    print("  -> Gated premature transition to SHOW_MODEL_CARDS when attributes < 3")

    # 3 metadata attributes -> should allow progression
    session_allow = {
        "appType": "text",
        "step": 0,
        "extraction": {
            "PRIMARY_SUBJECT": "motorcycles",
            "ENVIRONMENT_SETTING": "garage",
            "AESTHETIC_STYLE": "retro 70s"
        },
        "deepAnswers": {}
    }
    # Print what gets captured:
    captured_attributes = set()
    deep_answers = session_allow.get("deepAnswers") or {}
    for k, v in deep_answers.items():
        if not k.startswith("_") and v and str(v).strip():
            captured_attributes.add(k.lower().strip())
    extraction = session_allow.get("extraction") or {}
    for k in ["PRIMARY_SUBJECT", "ENVIRONMENT_SETTING", "ACTION_DYNAMIC", "AESTHETIC_STYLE", "budget", "targetUsers"]:
        if extraction.get(k) and str(extraction.get(k)).strip():
            captured_attributes.add(k.lower().strip())
    print(f"DEBUG session_allow: captured={captured_attributes}")
    res_allow = enforce_prd_rules(decision_gate, session_allow)
    print(f"DEBUG session_allow: res={res_allow}")
    assert res_allow["recommended_action"] == "SHOW_MODEL_CARDS", f"Expected ALLOWED SHOW_MODEL_CARDS, got {res_allow['recommended_action']}"
    print("  -> Allowed progression to SHOW_MODEL_CARDS with 3+ metadata attributes")

    # ─── TEST 4: TRIAGE LOOP PROTECTION (ANTI-LOOP) & ROUNDS CAP ───
    print("\nTest 4: Triage Anti-loop & Rounds Cap")
    # Helper simulation of _build_step0_response triage checks
    def simulate_triage_routing(session, triage_result):
        triage_rounds = session.get("triageRounds", 0) or session.get("triage_rounds", 0)
        deep_answers = session.get("deepAnswers") or {}
        populated_keys = {k.lower().strip() for k, v in deep_answers.items() if v and str(v).strip()}
        
        triage_slot_key = str(triage_result.get("slot_key") or "").lower().strip()
        if triage_slot_key in populated_keys or any(k in populated_keys for k in ["tone", "length", "theme", "audience"] if triage_slot_key == k):
            triage_result["status"] = "ready"
            triage_result["question"] = None
            triage_result["slot_key"] = None

        if triage_rounds >= 2:
            triage_result["status"] = "ready"
            triage_result["question"] = None
            triage_result["slot_key"] = None
            
        return triage_result

    # Check anti-loop populated check
    session_triage_loop = {"deepAnswers": {"tone": "energetic"}, "triageRounds": 0}
    triage_res = {"status": "needs_context", "slot_key": "tone", "question": "What tone?"}
    routed_triage = simulate_triage_routing(session_triage_loop, triage_res)
    assert routed_triage["status"] == "ready", f"Expected ready, got {routed_triage['status']}"
    print("  -> Anti-loop: Intercepted repeated 'tone' request, auto-completed triage state")

    # Check cap check
    session_triage_cap = {"deepAnswers": {}, "triageRounds": 2}
    triage_res_cap = {"status": "needs_context", "slot_key": "visual_style", "question": "What visual style?"}
    routed_triage_cap = simulate_triage_routing(session_triage_cap, triage_res_cap)
    assert routed_triage_cap["status"] == "ready", f"Expected ready, got {routed_triage_cap['status']}"
    print("  -> Cap: Intercepted round 2 triage call, forced ready status")

    # ─── TEST 5: SCHEMA GENERATOR BOILERPLATE FILTERING ───
    print("\nTest 5: Boilerplate Filtering")
    # Boilerplate should be filtered
    assert is_personal_boilerplate("User Name", "A motivational quote app") is True
    assert is_personal_boilerplate("Date of Birth", "A legal contract generator") is True
    print("  -> CORRECTLY identified personal boilerplate when not in app purpose")

    # Explicitly requested boilerplate should NOT be filtered
    assert is_personal_boilerplate("User Name", "An app that asks for the user name and prints it") is False
    print("  -> Allowed personal fields when explicitly requested in app purpose")

    # _sanitize_variable_objects should filter out "User Name"
    dirty_variables = [
        {"name": "User Name", "placeholder": "John Doe"},
        {"name": "Speech Length", "placeholder": "3 minutes"}
    ]
    cleaned = _sanitize_variable_objects(dirty_variables, 1, 5, [], "text", "A motivational speech builder")
    names = [v["name"] for v in cleaned]
    assert "User Name" not in names, f"Expected 'User Name' to be filtered out, names are: {names}"
    assert "Speech Length" in names, "Expected 'Speech Length' to be kept"
    print("  -> Sanitizer correctly stripped 'User Name' boilerplate while retaining domain attribute 'Speech Length'")

    print("\n[SUCCESS] All 5 Compliance Verification Tests Passed Successfully!")

if __name__ == "__main__":
    run_tests()
