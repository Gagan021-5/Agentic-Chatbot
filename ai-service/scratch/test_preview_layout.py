import sys
import os
import asyncio

# Add parent directory to path to import backend services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from routers.preview import test_preview, TestPreviewRequest, test_prompt, TestPromptRequest

import routers.preview

async def mock_fetch_pollinations(p, f):
    return "http://example.com/mock_image.png"

routers.preview._fetch_pollinations_with_fallback = mock_fetch_pollinations

class MockApp:
    class State:
        def __init__(self):
            self.llm = self
            self.vector_store = self
            self.has_groq = True
            self.has_openrouter = False
        
        async def search(self, query, categories, top_k):
            # If query contains style transfer keywords, mock matching the style transfer blueprint
            if "style transfer" in query.lower() or "artistic" in query.lower():
                return [{
                    "content": '{"tool_id": "style_transfer", "show_upload": true, "show_url_input": false, "layout_mode": "interactive", "config": {}}',
                    "relevance_score": 0.85
                }]
            return [{
                "content": '{"tool_id": "style_transfer", "show_upload": true, "show_url_input": false, "layout_mode": "interactive", "config": {}}',
                "relevance_score": 0.35  # Low score representing mismatch
            }]

        async def groq_chat(self, system_prompt, user_content, max_tokens):
            return {
                "choices": [{
                    "message": {
                        "content": "This is a mock generated script of over 300 words. " * 50
                    }
                }]
            }

        async def groq_completion(self, messages, model, response_format):
            import json
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "status": "ready",
                            "domain": "audio",
                            "confidence_score": 100,
                            "question": None,
                            "slots": []
                        })
                    }
                }]
            }
    
    def __init__(self):
        self.state = self.State()

class MockRequest:
    def __init__(self):
        self.app = MockApp()

async def run_tests():
    print("Running Preview Layout Resolution Tests...\n")
    
    # Test Case 1: Audio Meditation App (should not show upload UI)
    request = MockRequest()
    body_audio = TestPreviewRequest(
        appType="audio",
        variables={
            "Meditation Theme": "Stress relief",
            "Background Sound": "Rain or ocean waves"
        },
        systemPrompt="Generate a relaxing guided meditation script with ambient background sound."
    )
    
    response_audio = await test_preview(request, body_audio)
    ui_meta_audio = response_audio.ui_meta
    
    assert ui_meta_audio["show_upload"] is False, "Audio app should not show image upload option"
    assert ui_meta_audio["show_url_input"] is False, "Audio app should not show url input option"
    assert ui_meta_audio["layout_mode"] == "static", "Audio app layout mode should be static"
    print("[SUCCESS] Test Case 1: Audio meditation app successfully blocked image upload UI.")

    # Test Case 2: Image app with low-relevance style transfer match (should fall back and check heuristics)
    body_image_generic = TestPreviewRequest(
        appType="image",
        variables={
            "Subject": "A futuristic city"
        },
        systemPrompt="Render a beautiful high-quality futuristic landscape."
    )
    response_image_generic = await test_preview(request, body_image_generic)
    ui_meta_image_generic = response_image_generic.ui_meta
    
    assert ui_meta_image_generic["show_upload"] is True, "Image app should default to show upload via fallback"
    print("[SUCCESS] Test Case 2: Image app correctly defaults to show upload.")

    # Test Case 3: Image app with high-relevance style transfer blueprint match (should load style_transfer blueprint)
    body_style = TestPreviewRequest(
        appType="image",
        variables={
            "Style": "Anime filter",
            "Subject": "A portrait of a developer"
        },
        systemPrompt="Apply an artistic anime filter style transfer to the input photo."
    )
    response_style = await test_preview(request, body_style)
    ui_meta_style = response_style.ui_meta
    
    assert ui_meta_style["show_upload"] is True, "Style transfer app should show upload"
    assert ui_meta_style["active_tool"] == "style_transfer", "Style transfer tool should be active"
    print("[SUCCESS] Test Case 3: Style transfer image app successfully loaded style_transfer blueprint.")

    # Test Case 4: Edit App with Domain Shift (e.g. switching to music/song based app)
    # This should reset step to 0 (triage) and return the triage/gather requirements response
    from services.step_router import _exec_edit_app
    
    class MockSessionManager:
        async def save_session(self, session):
            pass

    class MockAppState:
        def __init__(self):
            self.llm = MockApp.State()
            self.session = MockSessionManager()

    app_state = MockAppState()
    session_shift = {
        "appType": "audio",
        "step": 2,
        "extraction": {
            "appPurpose": "Guided meditation audio app"
        },
        "dynamicContext": {"variables": []},
        "deepAnswers": {},
        "formConfirmed": True
    }
    decision_shift = {
        "extracted_variables": {
            "editInstruction": "Change: I want a song based app instead of meditation"
        }
    }
    
    res_shift = await _exec_edit_app(session_shift, "Change: I want a song based app instead of meditation", decision_shift, app_state)
    assert session_shift["step"] == 0, f"Expected step to reset to 0 (triage) due to domain shift, got {session_shift['step']}"
    print("[SUCCESS] Test Case 4: Domain shift successfully reset workflow step to 0.")

    # Test Case 5: Edit App without Domain Shift (e.g. changing language or tone of narration)
    # This should keep step = 2 (preview generation)
    session_no_shift = {
        "appType": "audio",
        "step": 2,
        "extraction": {
            "appPurpose": "Guided meditation audio app"
        },
        "dynamicContext": {"variables": []},
        "deepAnswers": {},
        "formConfirmed": True
    }
    decision_no_shift = {
        "extracted_variables": {
            "editInstruction": "Add soothing ocean breeze sound effects to background"
        }
    }
    
    res_no_shift = await _exec_edit_app(session_no_shift, "Add soothing ocean breeze sound effects to background", decision_no_shift, app_state)
    assert session_no_shift["step"] == 2, f"Expected step to stay as 2 (preview) for minor edits, got {session_no_shift['step']}"
    print("[SUCCESS] Test Case 5: Minor edit successfully kept workflow step at 2.")

    # Test Case 6: test_prompt variable substitution and robust cleanup
    prompt_history = []
    async def mock_openrouter_chat(system_prompt, user_content, model, temperature, max_tokens):
        prompt_history.append((system_prompt, user_content))
        return f"Processed: {user_content}"
    
    app_state.llm.openrouter_chat = mock_openrouter_chat
    
    body_prompt = TestPromptRequest(
        systemPrompt="Write a cinematic script.",
        userPrompt=(
            "I want to create a cinematic short video of a zombie apocalypse survival story. "
            "I'm looking for a **[Cinematic_Style]** visual mood, with a **[Main_Subject]** at the center "
            "- perhaps a **$$Survivor Name** navigating through the desolate streets of **$$Apocalypse Location**. "
            "Execute using my exact input Survivor name: $$survivor_name."
        ),
        testInputs={
            "survivor_name": "Alice",
            "apocalypse_location": "Atlanta"
        },
        modelHint="google/gemini-2.5-flash"
    )
    
    # Run the test prompt
    from routers.preview import TestPromptResponse
    response_prompt = await test_prompt(request, body_prompt)
    user_content_sent = response_prompt.output.replace("Processed: ", "")
    
    # 1. Substituted variables should be correctly resolved:
    # "Survivor Name" (with space) should be resolved to "Alice"
    # "Apocalypse Location" (with space) should be resolved to "Atlanta"
    # "$$survivor_name" (exact) should be resolved to "Alice"
    assert "**Alice**" in user_content_sent, f"Expected '**Alice**' in resolved prompt, got: {user_content_sent}"
    assert "**Atlanta**" in user_content_sent, f"Expected '**Atlanta**' in resolved prompt, got: {user_content_sent}"
    assert "Survivor name: Alice" in user_content_sent, f"Expected 'Survivor name: Alice' in resolved prompt, got: {user_content_sent}"
    
    # 2. Unsubstituted placeholders should be cleaned up cleanly, NOT leaving raw tags or garbled asterisks:
    # "[Cinematic_Style]" and "[Main_Subject]" should be completely removed
    assert "[Cinematic_Style]" not in user_content_sent, "Unsubstituted bracket variables should be cleaned up"
    assert "[Main_Subject]" not in user_content_sent, "Unsubstituted bracket variables should be cleaned up"
    assert "**" not in user_content_sent.replace("**Alice**", "").replace("**Atlanta**", ""), f"No stray empty bold markers should be left, got: {user_content_sent}"
    
    print("[SUCCESS] Test Case 6: test_prompt successfully resolved and cleaned up variables without garbled asterisks.")

    print("\n[SUCCESS] All Preview Layout Resolution Tests Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(run_tests())
