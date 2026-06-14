import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = json.loads((root / "data" / "models.json").read_text(encoding="utf-8"))
(root / "data" / "models.py").write_text(
    '"""AI model catalog — RentPrompts models.js."""\n\nMODELS = '
    + repr(data)
    + "\n",
    encoding="utf-8",
)

# Also update model_catalog.md in knowledge base
catalog_path = root / "knowledge" / "models" / "model_catalog.md"
if catalog_path.exists():
    catalog_text = catalog_path.read_text(encoding="utf-8")
    if "Pollinations Video" not in catalog_text:
        model_doc = """
## Pollinations Video
- Provider: Pollinations.ai
- Type: Video Generation
- Cost: 0.0 coins/run
- Tier: Fast
- Strengths: Fully dynamic video generation
- Prompt Style: Describe the motion and scene in detail.
- Best For: Real-time dynamic video preview
- Limitations: Short duration clips.
- Tags: free, fast, pollinations
"""
        catalog_path.write_text(catalog_text.rstrip() + "\n" + model_doc, encoding="utf-8")
