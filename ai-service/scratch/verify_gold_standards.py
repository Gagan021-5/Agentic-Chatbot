import sys
import os
import asyncio
import json
import re

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reconfigure stdout to use UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import get_settings
from rag.vector_store import VectorStoreManager
from services.prompt_generation import generate_seo
from services.llm import LLMService

async def test_all():
    print("==========================================")
    print("   RentPrompts ASO Pipeline Verification  ")
    print("==========================================")
    
    settings = get_settings()
    vs = VectorStoreManager(settings)
    await vs.initialize()
    
    print("\n--- 1. Testing RAG Priority Boosting ---")
    query = "creates ATS-friendly resumes for software engineering jobs"
    matches = await vs.search(
        query=query,
        categories=["examples"],
        top_k=5,
        boost_gold_standards=True
    )
    
    has_gold = False
    for idx, match in enumerate(matches):
        source = match.get("source")
        score = match.get("relevance_score")
        print(f"  [{idx+1}] Source: {source} | Score: {score}")
        if source == "marketplace_gold_standards.md":
            has_gold = True
            
    if has_gold:
        print("  [OK] RAG priority boosting successfully fetched and ranked marketplace_gold_standards.md!")
    else:
        print("  [FAIL] Did not retrieve marketplace_gold_standards.md in top results.")

    print("\n--- 2. Testing SEO Generation (LLM Workflow) ---")
    llm = LLMService()
    session = {
        "appType": "text",
        "extraction": {
            "appPurpose": "creates ATS-friendly resumes for software engineering jobs",
            "targetUsers": "students and job seekers",
            "detectedLanguage": "English"
        },
        "deepAnswers": {
            "industry": "technology"
        },
        "history": [
            {"role": "user", "content": "I want an app that creates ATS-friendly resumes for software engineering jobs."}
        ]
    }
    
    seo_res = await generate_seo(llm, session, vector_store=vs)
    print("  Generated ASO metadata:")
    print(json.dumps(seo_res, indent=4))
    
    print("\n--- 3. Enforcing Rules & Patterns Verification ---")
    app_name = seo_res.get("appName", "")
    desc = seo_res.get("appDescription", "")
    tags = seo_res.get("tags", [])
    
    # Rule 1: App Name
    print(f"  App Name: \"{app_name}\"")
    if 2 <= len(app_name.split()) <= 5 and len(app_name) <= 55:
        print("    [OK] App Name length and word count are within SaaS limits.")
    else:
        print("    [FAIL] App Name does not match 2-4 words guideline.")
        
    # Rule 2: Description
    print(f"  Description: \"{desc}\"")
    if len(desc) <= 150:
        print("    [OK] Description is under 150 characters.")
    else:
        print("    [FAIL] Description exceeds 150 characters.")
        
    # Rule 3: Tags
    print(f"  Tags: {tags}")
    if len(tags) == 7:
        print("    [OK] Exactly 7 tags generated.")
    else:
        print(f"    [FAIL] Expected exactly 7 tags, got {len(tags)}")
        
    invalid_tags = [t for t in tags if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", t)]
    if not invalid_tags:
        print("    [OK] All tags are in correct format (lowercase, hyphenated, no #).")
    else:
        print(f"    [FAIL] Invalid tags format: {invalid_tags}")

if __name__ == "__main__":
    asyncio.run(test_all())
