const mockData = {
  userBalance: 100000, // mock joules balance for demo
  market_data: [
    { category: "website", complexity: "simple", avg_hours: 4, floor_joules: 6000, market_joules: 10000 },
    { category: "website", complexity: "medium", avg_hours: 8, floor_joules: 12000, market_joules: 18000 },
    { category: "website", complexity: "complex", avg_hours: 14, floor_joules: 20000, market_joules: 28000 },
    { category: "mobile", complexity: "medium", avg_hours: 12, floor_joules: 18000, market_joules: 26000 },
    { category: "design", complexity: "simple", avg_hours: 3, floor_joules: 4000, market_joules: 8000 },
    { category: "design", complexity: "medium", avg_hours: 6, floor_joules: 8000, market_joules: 14000 },
    { category: "ai-app", complexity: "simple", avg_hours: 3, floor_joules: 4000, market_joules: 8000 },
    { category: "ai-app", complexity: "medium", avg_hours: 6, floor_joules: 8000, market_joules: 14000 },
    { category: "ai-app", complexity: "complex", avg_hours: 10, floor_joules: 14000, market_joules: 22000 }
  ]
};

export default mockData;
