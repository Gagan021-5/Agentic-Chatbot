import MODELS from "./models.js";

const PLATFORM_FEE = 500;

function getModelCost(modelId, appType) {
  const list = MODELS[appType] || [];
  const model = list.find((m) => m.id === modelId);
  return model ? model.cost : 0;
}

function buildBudgetTiers(mockRow) {
  return {
    lean: {
      joules: mockRow.floor_joules + PLATFORM_FEE,
      label: "Lean MVP",
      desc: "Core features only",
      usd: ((mockRow.floor_joules + PLATFORM_FEE) * 0.0108).toFixed(2)
    },
    recommended: {
      joules: mockRow.market_joules + PLATFORM_FEE,
      label: "Recommended",
      desc: "Best balance of quality and speed",
      usd: ((mockRow.market_joules + PLATFORM_FEE) * 0.0108).toFixed(2)
    },
    full: {
      joules: Math.round(mockRow.market_joules * 1.15) + PLATFORM_FEE,
      label: "Full Scope",
      desc: "Maximum quality with buffer",
      usd: ((Math.round(mockRow.market_joules * 1.15) + PLATFORM_FEE) * 0.0108).toFixed(2)
    }
  };
}

export { getModelCost, buildBudgetTiers };
