"""Cost calculator — RentPrompts costCalculator.js."""

from data.models import MODELS

PLATFORM_FEE = 500


def get_model_cost(model_id: str, app_type: str) -> float:
    models = MODELS.get(app_type, [])
    for model in models:
        if model.get("id") == model_id:
            return model.get("cost", 0)
    return 0


def build_budget_tiers(mock_row: dict) -> dict:
    floor_j = mock_row["floor_joules"]
    market_j = mock_row["market_joules"]
    return {
        "lean": {
            "joules": floor_j + PLATFORM_FEE,
            "label": "Lean MVP",
            "desc": "Core features only",
            "usd": f"{(floor_j + PLATFORM_FEE) * 0.0108:.2f}",
        },
        "recommended": {
            "joules": market_j + PLATFORM_FEE,
            "label": "Recommended",
            "desc": "Best balance of quality and speed",
            "usd": f"{(market_j + PLATFORM_FEE) * 0.0108:.2f}",
        },
        "full": {
            "joules": round(market_j * 1.15) + PLATFORM_FEE,
            "label": "Full Scope",
            "desc": "Maximum quality with buffer",
            "usd": f"{(round(market_j * 1.15) + PLATFORM_FEE) * 0.0108:.2f}",
        },
    }
