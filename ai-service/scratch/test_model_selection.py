import sys
import os
import asyncio
import traceback

# Reconfigure stdout/stderr to handle Unicode characters (emojis) safely on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add ai-service to path
sys.path.append(r"c:\Users\Gagan\Desktop\Chatbot Rentprompts\rentprompts-agent\ai-service")

from services.step_router import _exec_generate_preview_state
from services.llm import LLMService

class MockAppState:
    def __init__(self):
        self.llm = LLMService()
        self.vector_store = None

async def main():
    app_state = MockAppState()
    
    # Let's mock a session state just like the one in the user's conversation
    state = {
        "sessionId": "test-session-123",
        "history": [
            {"role": "user", "content": "I want an app that writes personalized cold emails for sales reps"},
            {"role": "agent", "content": "What is the primary goal of these cold emails, such as scheduling a meeting, introducing a product, or following up on a lead?"},
            {"role": "user", "content": "targeting SaaS companies in India"},
            {"role": "agent", "content": "Excellent choice! I have saved the blueprint for your app. One last parameter to proceed: What is your budget preference per run?"},
            {"role": "user", "content": "Medium (5 - 20 coins)"}
        ],
        "extraction": {
            "appType": "text",
            "appPurpose": "writes personalized cold emails for sales reps targeting SaaS companies in India",
            "targetUsers": "sales reps",
            "budget": "Medium (5 - 20 coins)",
            "wantsImageInput": False,
            "detectedLanguage": "English",
            "userTone": "professional",
            "oneLineUnderstanding": "writes personalized cold emails for sales reps targeting SaaS companies in India"
        },
        "deep_answers": {
            "email_purpose": "introducing a product",
            "personalization_level": "referencing specific company details"
        },
        "prompt_data": {},
        "seo_data": {},
        "triage_rounds": 1,
        "current_step": 1,
        "app_type": "text",
        "model_id": "o4-mini",
        "model_name": "O4 Mini",
        "model_cost": 6.0,
        "verification_metadata": {
            "app_type": "explicit",
            "ingestion_vector": "explicit",
            "budget": "explicit"
        }
    }
    
    # Try selecting O4 Mini
    print("Testing select o4-mini:")
    try:
        # We need to bypass the try/except block inside _exec_generate_preview_state if we want the raw traceback.
        # But let's first run it directly to see if it snags.
        res = await _exec_generate_preview_state(state, "select o4-mini", app_state)
        print(f"Result reply: {res.get('reply')}")
    except Exception as e:
        print("Raw exception caught:")
        traceback.print_exc()

    await app_state.llm.close()

if __name__ == "__main__":
    asyncio.run(main())
