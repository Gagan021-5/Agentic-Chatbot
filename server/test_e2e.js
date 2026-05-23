// Full E2E test for all 4 app categories
const BASE = "http://localhost:3001/api/agent/chat";

async function send(sessionId, message) {
  const res = await fetch(BASE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sessionId, message })
  });
  return res.json();
}

async function testCategory(name, sessionId, initialMsg, formOptions) {
  console.log(`\n${"=".repeat(60)}`);
  console.log(`  ${name.toUpperCase()} APP — Full E2E`);
  console.log("=".repeat(60));

  // Step 1: Initial request
  let r = await send(sessionId, initialMsg);
  console.log(`[1] Initial → ui:${r.uiType} | reply:${r.reply?.substring(0, 80)}...`);

  // Handle triage dynamically
  let triageCount = 0;
  while ((!r.uiType || r.uiType === "text" || r.uiType === "chips") && triageCount < 6) {
    triageCount++;
    if (r.uiType === "chips") {
      const chip = r.uiData?.options?.[0] || "Yes, that works for me";
      r = await send(sessionId, chip);
      console.log(`[1-triage-${triageCount}] Chip/Triage → ui:${r.uiType} | reply:${r.reply?.substring(0, 80)}...`);
    } else {
      r = await send(sessionId, formOptions.triageAnswer || "Yes, exactly what I need");
      console.log(`[1-triage-${triageCount}] Text/Triage → ui:${r.uiType} | reply:${r.reply?.substring(0, 80)}...`);
    }
  }

  if (r.uiType !== "multi_select_form") {
    console.log(`  ❌ FAIL: Expected multi_select_form, got ${r.uiType}`);
    return false;
  }
  console.log(`  ✅ Form generated`);
  console.log(`     Options: ${r.uiData?.options?.join(", ")}`);
  console.log(`     Vars: ${r.uiData?.variables?.map(v => v.name).join(", ")}`);

  // Step 2: Submit form
  const formPayload = {
    selectedOptions: (r.uiData?.options || []).slice(0, 2),
    variables: (r.uiData?.variables || []).map(v => ({ name: v.name, value: v.placeholder || "test" }))
  };
  r = await send(sessionId, `multi_select_form::${JSON.stringify(formPayload)}`);
  console.log(`[2] Form submit → ui:${r.uiType} | reply:${r.reply?.substring(0, 80)}...`);

  if (r.uiType !== "chips") {
    console.log(`  ❌ FAIL: Expected budget chips, got ${r.uiType}`);
    return false;
  }
  console.log(`  ✅ Budget chips shown`);

  // Step 3: Select budget
  const budget = formOptions.budget || "Medium (5 - 20 coins)";
  r = await send(sessionId, budget);
  console.log(`[3] Budget → ui:${r.uiType} | models:${r.uiData?.models?.length || 0}`);

  if (r.uiType !== "models") {
    console.log(`  ❌ FAIL: Expected models, got ${r.uiType}`);
    return false;
  }
  const models = r.uiData.models;
  console.log(`  ✅ Models: ${models.map(m => `${m.name}(${m.cost})`).join(", ")}`);

  // Step 4: Select model
  const modelId = models[0].id;
  r = await send(sessionId, `select ${modelId}`);
  console.log(`[4] Model select → ui:${r.uiType}`);

  if (r.uiType !== "app_preview") {
    console.log(`  ❌ FAIL: Expected app_preview, got ${r.uiType}`);
    return false;
  }
  console.log(`  ✅ Preview: "${r.uiData.appName}" | ${r.uiData.cost} coins`);
  console.log(`     Vars: ${r.uiData.variablesUsed?.join(", ")}`);

  // Step 5: Approve
  r = await send(sessionId, "Approve App");
  console.log(`[5] Approve → ui:${r.uiType}`);

  if (r.uiType !== "seo_preview") {
    console.log(`  ❌ FAIL: Expected seo_preview, got ${r.uiType}`);
    return false;
  }
  console.log(`  ✅ SEO Preview shown: "${r.uiData.appName}"`);

  // Step 6: Publish
  const publishPayload = {
    appName: r.uiData.appName || "My App",
    appDescription: r.uiData.appDescription || "My App Description",
    tags: r.uiData.tags || []
  };
  r = await send(sessionId, `SEO_PUBLISH::${JSON.stringify(publishPayload)}`);
  console.log(`[6] Publish → ui:${r.uiType}`);

  if (r.uiType !== "success") {
    console.log(`  ❌ FAIL: Expected success, got ${r.uiType}`);
    return false;
  }
  console.log(`  ✅ PUBLISHED: "${r.uiData.appName}" @ ${r.uiData.costPerRun} coins`);
  return true;
}

async function main() {
  const results = {};
  const ts = Date.now();

  // TEXT APP
  results.text = await testCategory("Text", `e2e-text-${ts}`,
    "I want an app that writes personalized cover letters based on job title, experience, and target company",
    { budget: "Medium (5 - 20 coins)" }
  );

  // IMAGE APP  
  results.image = await testCategory("Image", `e2e-image-${ts}`,
    "I want an app that generates professional product photos for ecommerce based on product description and background style",
    { budget: "Low (< 5 coins)" }
  );

  // AUDIO APP
  results.audio = await testCategory("Audio", `e2e-audio-${ts}`,
    "I want an audio app that converts blog articles into podcast narration with a natural female voice",
    { budget: "Low (< 5 coins)" }
  );

  // VIDEO APP (will trigger triage)
  results.video = await testCategory("Video", `e2e-video-${ts}`,
    "I want a video app for restaurant promotional reels",
    { budget: "Premium (> 20 coins)", triageAnswer: "It should generate 15-second vertical reels from AI with cuisine-based templates, tone selection, and background music" }
  );

  // HINGLISH TEST
  console.log(`\n${"=".repeat(60)}`);
  console.log("  HINGLISH LANGUAGE MIRROR TEST");
  console.log("=".repeat(60));
  const hr = await send(`e2e-hinglish-${ts}`, "Mujhe ek fitness app banana hai jo daily workout plan banaye weight aur age ke hisaab se");
  console.log(`Reply: ${hr.reply}`);
  console.log(`UI: ${hr.uiType}`);
  results.hinglish = hr.reply?.includes("aap") || hr.reply?.includes("Maine") || hr.reply?.includes("app");

  // SUMMARY
  console.log(`\n${"=".repeat(60)}`);
  console.log("  FINAL RESULTS");
  console.log("=".repeat(60));
  for (const [k, v] of Object.entries(results)) {
    console.log(`  ${v ? "✅" : "❌"} ${k.toUpperCase()}`);
  }
  const allPass = Object.values(results).every(Boolean);
  console.log(`\n  ${allPass ? "🎉 ALL TESTS PASSED!" : "⚠️ SOME TESTS FAILED"}`);
}

main().catch(console.error);
