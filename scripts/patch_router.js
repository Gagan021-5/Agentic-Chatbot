import fs from "fs";
import path from "path";
let file = path.join(__dirname, '..', 'server', 'lib', 'stepRouter.js');
let content = fs.readFileSync(file, 'utf8');

// 1. rankModels Replacement
const oldRankModelsRegex = /function rankModels\(models, userMessage, budgetStr\) \{[\s\S]*?\n\}/;
const newRankModels = `function rankModels(availableModels, userInput, budgetStr) {
  if (!availableModels || availableModels.length === 0) return [];
  
  let filtered = [...availableModels];

  // 1. STRICT BUDGET FILTERING (Using our established ranges)
  const b = (budgetStr || "").toLowerCase();
  if (b) {
    if (b.includes("medium") || b.includes("5-20") || b.includes("5 - 20")) {
      filtered = filtered.filter(m => m.cost >= 5 && m.cost <= 20);
    } else if (b.includes("low") || b.includes("under 5") || b.includes("< 5")) {
      filtered = filtered.filter(m => m.cost > 0 && m.cost < 5);
    } else if (b.includes("premium") || b.includes("best") || b.includes("> 20")) {
      filtered = filtered.filter(m => m.cost >= 20);
    } else if (b.includes("free") || b.includes("0 coins")) {
      filtered = filtered.filter(m => m.cost === 0);
    } else {
      const numberMatch = b.match(/\\d+(\\.\\d+)?/);
      if (numberMatch) {
        filtered = filtered.filter(m => m.cost <= parseFloat(numberMatch[0]));
      }
    }
  }

  // UNMATCHABLE BUDGET GUARD (If budget is too strict, return absolute cheapest)
  if (filtered.length === 0) {
    return [...availableModels].sort((a, b) => a.cost - b.cost).slice(0, 3);
  }

  // 2. SCORING ENGINE
  const input = (userInput || "").toLowerCase();
  
  const scoredModels = filtered.map(model => {
    let score = 0;
    
    // Boost score if the user's prompt contains the model's tags
    if (model.tags) {
      model.tags.forEach(tag => {
        if (input.includes(tag.toLowerCase())) score += 5;
      });
    }

    // Boost score based on Tiers matching user intent
    if (input.includes("fast") || input.includes("quick") || input.includes("speed")) {
      if (model.tier === "fast") score += 5;
    }
    if (input.includes("quality") || input.includes("best") || input.includes("advanced")) {
      if (model.tier === "premium" || model.tier === "ultra") score += 5;
    }
    if (input.includes("cheap") || input.includes("affordable")) {
      // Reward lower cost models
      score += (20 - model.cost); 
    }

    return { ...model, score };
  });

  // 3. SORT & SELECT
  // Sort primarily by score (highest first). If scores tie, sort by cost (cheapest first).
  scoredModels.sort((a, b) => b.score - a.score || a.cost - b.cost);

  // Return exactly the top 3 (removing the temporary score property to keep data clean)
  return scoredModels.slice(0, 3).map(({ score, ...rest }) => rest);
}`;
content = content.replace(oldRankModelsRegex, newRankModels);

// 2. Cost Locking
content = content.replace(/cost: seoData\.suggestedPrice \|\| session\.modelCost,/g, 'cost: session.modelCost, // Strictly overrides any LLM hallucinations or math errors');
content = content.replace(/costPerRun: session\.seoData\?\.suggestedPrice \|\| session\.modelCost,/g, 'costPerRun: session.modelCost,');

// 3. showModels budget fallback warning
const oldShowModels = `// FIXED: Removed ConfirmCard, set step=1
async function showModels(session) {
  const fullText = [
    session.extraction?.appPurpose || '',
    session.extraction?.oneLineUnderstanding || '',
    JSON.stringify(session.deepAnswers || {})
  ].join(' ');
  // UPDATE: Pull budget from deep answers first, fallback to extraction
  const budget = session.deepAnswers?.budgetPreference || session.extraction?.budget;
  
  const models = rankModels(MODELS[session.appType] || [], fullText, budget);
  
  session.step = 1; 
  session.awaitingConfirmation = false;
  await saveSession(session);
  
  return {
    reply: \`Here are the top 3 models for your \${session.appType} app. Click a model card below to select it:\`,
    uiType: 'models',
    uiData: { appType: session.appType, models },
    nextStep: 1,
    coins: null
  };
}`;
const newShowModels = `// FIXED: Removed ConfirmCard, set step=1
async function showModels(session) {
  const fullText = [
    session.extraction?.appPurpose || '',
    session.extraction?.oneLineUnderstanding || '',
    JSON.stringify(session.deepAnswers || {})
  ].join(' ');
  
  const budget = session.deepAnswers?.budgetPreference || session.extraction?.budget || "";
  const models = rankModels(MODELS[session.appType] || [], fullText, budget);
  
  session.step = 1; 
  session.awaitingConfirmation = false;
  await saveSession(session);

  let replyText = \`Here are the top 3 models for your \${session.appType} app. Click a model card below to select it:\`;
  if (budget && models.length > 0) {
    const numberMatch = budget.toLowerCase().match(/\\d+(\\.\\d+)?/);
    if (numberMatch && models[0].cost > parseFloat(numberMatch[0])) {
      replyText = \`I couldn't find any models under \${numberMatch[0]} coins for this specific task, but here are the absolute cheapest options available:\`;
    }
  }

  return {
    reply: replyText,
    uiType: 'models',
    uiData: { appType: session.appType, models },
    nextStep: 1,
    coins: null
  };
}`;
content = content.replace(oldShowModels, newShowModels);

// 4. isGreeting and isGibberish
const oldGibberishBlockRegex = /\/\/ 2\. Abuse \/ gibberish guard[\s\S]*?if \(!text \|\| text === ''\) \{/;
const newGibberishBlock = `const trimmedText = text.trim();
  
  // 1. Pure Greeting Interceptor
  // Catches "hello", "hi", "hey", "hy", "greetings" even with punctuation like "hello!!"
  const isGreeting = /^(hi|hello|hey|hy|hola|greetings)[\\s!\\.]*$/i.test(trimmedText);
  
  if (isGreeting) {
    return {
      reply: "Hello! I am ready to help you build your AI application today. To get started, what type of output does your app need?",
      uiType: 'chips',
      uiData: { options: ['Text', 'Image', 'Audio', 'Video', 'Vision'] },
      nextStep: session.step,
      coins: null
    };
  }

  // 2. Abuse / gibberish guard
  const symbolCount = (trimmedText.match(/[^a-zA-Z0-9\\s]/g) || []).length;
  
  const isGibberish = 
    trimmedText.length < 2 || // Too short
    /(asdf|qwer|zxcv|hjkl|asdasd)/i.test(trimmedText) || // Expanded keyboard smash
    /[a-zA-Z0-9]{20,}/.test(trimmedText) || // Huge 20+ char block with no spaces
    (symbolCount > trimmedText.length / 2 && trimmedText.length > 5) || // Over 50% symbols
    /[bcdfghjklmnpqrstvwxz]{5,}/i.test(trimmedText) || // NEW: 5+ consonants in a row (catches 'abcszs')
    /(.)\\1{4,}/i.test(trimmedText) || // 5+ of the exact same character
    (!trimmedText.includes(' ') && trimmedText.length > 8 && /[0-9@#\\$\\%\\^\\&\\*]/.test(trimmedText)); // NEW: 8+ chars, no spaces, mixed with numbers/symbols (catches 'asdasd@#q31')

  if (isGibberish) {
    return {
      // Changed the reply to be exactly what you want when nonsense is typed
      reply: \`Sorry, I didn't quite get what you want to build today. What type of output does your app need?\`,
      uiType: 'chips',
      uiData: { options: ['Text', 'Image', 'Audio', 'Video', 'Vision'] },
      nextStep: session.step,
      coins: null
    };
  }

  if (!text || text === '') {`;

content = content.replace(oldGibberishBlockRegex, newGibberishBlock);

fs.writeFileSync(file, content);
console.log('Restored correctly');
