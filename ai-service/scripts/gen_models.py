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
