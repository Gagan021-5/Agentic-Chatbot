
const OFF_TOPIC_KEYWORDS = [
  // science/general knowledge (these only fire when there's NO active session)
  'gravity','physics','chemistry','biology','history',
  'math','equation','formula','theorem','law of',
  'what is','who is','who was','when did','where is',
  'capital of','population of','how far','how old',
  // weather/news
  'weather','news','stock','price of bitcoin',
  'temperature','forecast',
  // coding help unrelated
  'fix my code','debug','error in my code',
  'how to install','npm install','python',
  // personal/emotional
  'i am sad','i am happy','i love you','you are',
  'are you human','are you ai','who made you',
  // harmful
  'hack','crack','illegal','kill','weapon'
  // NOTE: 'explain', 'define', 'meaning of', 'tell me about' intentionally removed
  // — these are common triage answers (e.g. "explain a section", "define a term")
];

const QUESTION_STARTERS = [
  'what is the','what are the','who is','who was',
  'when is','when was','where is','how does gravity',
  'explain the','define the','tell me about',
  'what causes','why does the sun'
];

// Words that indicate the message is about app creation
// and should NOT be treated as off-topic even if it
// contains a keyword match
const APP_CONTEXT_WORDS = [
  'app','build','create','make','generate','tool',
  'rentprompts','marketplace','publish','image app',
  'video app','text app','audio app','vision app',
  'i want','i need','my app','our app','for my',
  'for our','product','clinic','company','business',
  'generator','creator','builder'
];

export function isOffTopic(message, session) {
  const msg = message.toLowerCase().trim();
  if (msg.length < 4) return false;

  // NEVER fire off-topic detection mid-conversation
  // If the session has history or active triage, the user is answering our questions
  if (session) {
    const hasHistory = Array.isArray(session.history) && session.history.length > 1;
    const hasTriage  = (session.triageRounds || 0) > 0 || session.awaitingTriageAnswer === true;
    const hasContext = session.dynamicContext || session.appType;
    if (hasHistory || hasTriage || hasContext) return false;
  }

  // If the message contains app-creation context words,
  // it's NOT off-topic even if it contains a keyword match
  const hasAppContext = APP_CONTEXT_WORDS.some(w => msg.includes(w));
  if (hasAppContext) return false;

  for (const s of QUESTION_STARTERS) {
    if (msg.startsWith(s)) return true;
  }
  for (const kw of OFF_TOPIC_KEYWORDS) {
    if (msg.includes(kw)) return true;
  }
  return false;
}

export const OFF_TOPIC_RESPONSE = {
  reply: `I'm RentPrompts Agent — I help you create and publish AI-powered apps on RentPrompts marketplace.\n\nI can help you build apps that generate:\n- 🖼️ Images — portraits, art, product photos\n- 🎥 Videos — animations, cinematic clips, reels\n- 📝 Text — blogs, emails, scripts, stories\n- 🔊 Audio — voiceovers, music, speech\n- 👁️ Vision — image analysis, object detection\n\nWhat kind of AI app would you like to create?`,
  uiType: 'chips',
  uiData: { options: ['Image app','Video app','Text app','Audio app','Vision app'] },
  nextStep: 0,
  coins: null
};
