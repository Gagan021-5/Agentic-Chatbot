
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

// ═══════════════════════════════════════════════════
// REQUIRED FIELDS — DEEP first, COMMON after
// Conditional fields supported via `condition` fn
// ═══════════════════════════════════════════════════

export const REQUIRED_FIELDS = {

  image: [
    // DEEP FIRST — specific to what user described
    {
      field: 'imageTransformationType',
      question: 'What kind of transformation should ' +
                'this app do to the image?',
      condition: (req) => req.appPurpose?.includes('convert') ||
                          req.appPurpose?.includes('transform') ||
                          req.appPurpose?.includes('into') ||
                          req.appPurpose?.includes('superhero'),
      chips: [
        'Transform person into a character/style',
        'Change art style of the image',
        'Add elements to existing image',
        'Enhance or restore image quality',
        'Generate completely new from description',
        'Something else'
      ]
    },
    {
      field: 'imageStyle',
      question: 'What visual style should the ' +
                'output image have?',
      chips: [
        'Comic book / Marvel style',
        'Anime / manga style',
        'Cinematic / movie poster style',
        'Photorealistic with enhancements',
        'Oil painting / artistic',
        'Cyberpunk / sci-fi style',
        'Dark fantasy / dramatic',
        'Cartoon / illustrated',
        'User chooses the style'
      ]
    },
    {
      field: 'imageMood',
      question: 'What mood or energy should the ' +
                'generated images have?',
      chips: [
        'Powerful and heroic',
        'Dark and dramatic',
        'Bright and vibrant',
        'Epic and cinematic',
        'Fun and playful',
        'Professional and clean',
        'User controls the mood'
      ]
    },
    {
      field: 'imageInputType',
      question: 'What does the user upload or provide ' +
                'to generate the image?',
      chips: [
        'Upload a photo of themselves',
        'Upload any reference image',
        'Type a text description only',
        'Both photo upload and text prompt',
        'Multiple images as reference'
      ]
    },
    {
      field: 'imageOutputDetails',
      question: 'Any specific details about the output? ' +
                'For example costume style, background, ' +
                'or framing?',
      chips: [
        'Full body portrait with costume',
        'Face and upper body only',
        'Action pose with effects',
        'Simple headshot transformation',
        'Custom scene and background',
        'User describes what they want'
      ]
    },
    // COMMON QUESTIONS COME AFTER DEEP ONES
    {
      field: 'targetUsers',
      question: 'Who is the main audience for this app?',
      chips: [
        'General public / anyone',
        'Kids and families',
        'Gaming community',
        'Social media users',
        'Content creators',
        'Businesses / brands',
        'My internal team'
      ]
    },
    {
      field: 'primaryUseCase',
      question: 'How will people mainly use this app?',
      chips: [
        'Just for fun / personal use',
        'Social media profile pictures',
        'Gaming avatars and characters',
        'Marketing and promotions',
        'Gifts and merchandise',
        'Brand mascot creation',
        'Something else'
      ]
    }
  ],

  video: [
    {
      field: 'videoType',
      question: 'What kind of video should this app create?',
      chips: [
        'Animate a still photo into video',
        'Text description to video',
        'Transform video style',
        'Create cinematic scenes',
        'Short social media reels',
        'Product showcase videos',
        'Avatar / talking head'
      ]
    },
    {
      field: 'videoEffects',
      question: 'What motion or visual effects ' +
                'does the video need?',
      chips: [
        'Smooth cinematic camera movement',
        'Dynamic action sequences',
        'Slow motion dramatic effect',
        'Zoom and parallax depth',
        'Particle and light effects',
        'Natural realistic motion',
        'User chooses the effect'
      ]
    },
    {
      field: 'videoDuration',
      question: 'How long should each generated ' +
                'video clip be?',
      chips: [
        '3-5 seconds (short clip)',
        '5-10 seconds (medium)',
        '10-30 seconds (longer)',
        'User sets the duration'
      ]
    },
    {
      field: 'videoAudio',
      question: 'Should the video include sound?',
      chips: [
        'Yes — AI generated sound effects',
        'Yes — background music',
        'Silent — no audio',
        'User chooses'
      ]
    },
    {
      field: 'targetUsers',
      question: 'Who is the main audience for this app?',
      chips: [
        'Content creators / influencers',
        'Marketing teams',
        'General public',
        'Businesses',
        'Filmmakers',
        'My internal team'
      ]
    },
    {
      field: 'primaryUseCase',
      question: 'What will people mainly use it for?',
      chips: [
        'Social media content',
        'Marketing campaigns',
        'Personal creative projects',
        'Product showcases',
        'Entertainment',
        'Business presentations'
      ]
    }
  ],

  text: [
    {
      field: 'textOutputType',
      question: 'What kind of text should this app write?',
      chips: [
        'Blog posts and articles',
        'Social media captions',
        'Email and newsletters',
        'Product descriptions',
        'Scripts and screenplays',
        'Stories and creative writing',
        'Reports and summaries',
        'Code and technical docs'
      ]
    },
    {
      field: 'textTone',
      question: 'What tone should the writing have?',
      chips: [
        'Professional and formal',
        'Casual and conversational',
        'Persuasive and sales-focused',
        'Educational and clear',
        'Creative and expressive',
        'Technical and precise',
        'User controls the tone'
      ]
    },
    {
      field: 'textLength',
      question: 'How long should the generated content be?',
      chips: [
        'Short — under 100 words',
        'Medium — 100 to 500 words',
        'Long — 500 words or more',
        'User sets the length'
      ]
    },
    {
      field: 'textInputType',
      question: 'What does the user provide ' +
                'to generate the text?',
      chips: [
        'A topic or keyword',
        'A rough draft to improve',
        'A URL or document to summarize',
        'Just a few instructions',
        'A template with blanks to fill'
      ]
    },
    {
      field: 'targetUsers',
      question: 'Who will use this writing app?',
      chips: [
        'Bloggers and writers',
        'Marketing teams',
        'Students',
        'Business professionals',
        'Content creators',
        'General public'
      ]
    },
    {
      field: 'primaryUseCase',
      question: 'What is the main reason they use it?',
      chips: [
        'Save time writing',
        'Overcome writers block',
        'Scale content production',
        'Improve writing quality',
        'Learn and practice writing',
        'Automate repetitive writing'
      ]
    }
  ],

  audio: [
    {
      field: 'audioType',
      question: 'What kind of audio should ' +
                'this app generate?',
      chips: [
        'Voice narration / text to speech',
        'AI music generation',
        'Sound effects',
        'Podcast production',
        'Voice cloning',
        'Speech to text transcription',
        'Language dubbing'
      ]
    },
    {
      field: 'audioVoiceStyle',
      question: 'What voice or audio style is needed?',
      chips: [
        'Professional narrator voice',
        'Warm conversational tone',
        'Dramatic / storytelling voice',
        'Multiple language support',
        'Character voices',
        'User picks from voice options'
      ]
    },
    {
      field: 'audioDuration',
      question: 'How long is the typical audio output?',
      chips: [
        'Short clips under 30 seconds',
        '30 seconds to 2 minutes',
        '2 to 10 minutes',
        'Long form 10+ minutes',
        'User controls the length'
      ]
    },
    {
      field: 'targetUsers',
      question: 'Who is this audio app built for?',
      chips: [
        'Podcasters',
        'Content creators',
        'Businesses',
        'Educators',
        'General public',
        'Developers building products'
      ]
    },
    {
      field: 'primaryUseCase',
      question: 'What will they mainly use it for?',
      chips: [
        'Voiceovers for videos',
        'Podcast production',
        'Accessibility features',
        'Language learning',
        'Marketing audio content',
        'Personal entertainment'
      ]
    }
  ],

  vision: [
    {
      field: 'visionTask',
      question: 'What should this app do when ' +
                'it sees an image?',
      chips: [
        'Describe what is in the image',
        'Detect and label objects',
        'Read text from image (OCR)',
        'Analyze medical or scientific images',
        'Inspect product quality',
        'Understand documents and forms',
        'Answer questions about an image',
        'Detect faces or emotions'
      ]
    },
    {
      field: 'visionOutputFormat',
      question: 'What format should the analysis ' +
                'result be in?',
      chips: [
        'Plain text description',
        'Structured report with sections',
        'JSON data for developers',
        'Simple yes/no answer',
        'Confidence score with labels',
        'User-friendly summary'
      ]
    },
    {
      field: 'visionAccuracy',
      question: 'How critical is accuracy for ' +
                'this use case?',
      chips: [
        'Very high — medical or legal use',
        'High — professional business use',
        'Medium — general consumer use',
        'Flexible — creative or fun use'
      ]
    },
    {
      field: 'targetUsers',
      question: 'Who will use this vision app?',
      chips: [
        'Healthcare professionals',
        'Business operations teams',
        'Developers building products',
        'Researchers',
        'General consumers',
        'Quality control teams'
      ]
    },
    {
      field: 'primaryUseCase',
      question: 'What problem does this app solve for them?',
      chips: [
        'Save time on manual inspection',
        'Extract information from images',
        'Make content accessible',
        'Automate quality checks',
        'Enable visual search',
        'Assist people with visual needs'
      ]
    }
  ]
};

export const ENTERPRISE_EXTRA = [
  {
    field: 'teamSize',
    question: 'How many people will use this app?',
    chips: ['Just me','2-10 people','10-50 people','50-200 people','200+ people / Enterprise']
  },
  {
    field: 'integrationNeeds',
    question: 'Does this app need to connect to any existing tools or systems?',
    chips: [
      'No, standalone app','Yes, connect to our website',
      'Yes, integrate with Slack/Teams','Yes, connect to our CRM/database',
      'Yes, REST API access needed','Not sure yet'
    ]
  },
  {
    field: 'volumeExpectation',
    question: 'How many times per day will this app be used?',
    chips: ['Under 100 runs/day','100-1000 runs/day','1000-10000 runs/day','10000+ runs/day','Not sure yet']
  }
];

// ═══════════════════════════════════════════════════
// REQUIREMENT GATHERING HELPERS
// ═══════════════════════════════════════════════════

export function getRequiredFields(session) {
  const appFields = REQUIRED_FIELDS[session.appType] || [];
  const base = [...appFields];
  if (session.enterpriseSignals) base.push(...ENTERPRISE_EXTRA);
  return base;
}

export function getNextQuestion(session) {
  const appFields = REQUIRED_FIELDS[session.appType] || [];
  // Include enterprise extras if applicable
  const allFields = session.enterpriseSignals
    ? [...appFields, ...ENTERPRISE_EXTRA]
    : appFields;

  for (const f of allFields) {
    // skip if already answered
    if (session.requirements && session.requirements[f.field]) continue;

    // skip if condition exists and is not met
    if (f.condition && !f.condition(session.requirements || {})) {
      continue;
    }

    // this is the next question to ask
    return f;
  }

  return null; // all done
}

export function saveAnswer(session, fieldName, answer) {
  if (!session.requirements) session.requirements = {};
  session.requirements[fieldName] = answer;
}

export function getProgressText(session) {
  const fields = getRequiredFields(session);
  // Count only non-conditional or condition-met fields for accurate progress
  const applicableFields = fields.filter(f => {
    if (f.condition && !f.condition(session.requirements || {})) return false;
    return true;
  });
  const collected = applicableFields.filter(f => session.requirements && session.requirements[f.field]).length;
  return `(${collected}/${applicableFields.length})`;
}

export function buildRequirementsSummary(session) {
  const r = session.requirements || {};
  return Object.entries(r).map(([k, v]) => `• ${k}: ${v}`).join('\n');
}

// ═══════════════════════════════════════════════════
// SMART PRE-FILL FROM FIRST MESSAGE
// Reads extraction and pre-fills as many requirement
// fields as possible so agent never asks what it
// already knows
// ═══════════════════════════════════════════════════

export function preFillFromExtraction(session) {
  const e = session.extraction;
  const msg = (e?.oneLineUnderstanding || '').toLowerCase();
  const purpose = (e?.appPurpose || '').toLowerCase();

  if (!session.requirements) session.requirements = {};
  const r = session.requirements;

  // Pre-fill appPurpose if extracted
  if (e?.appPurpose && !r.appPurpose) {
    r.appPurpose = e.appPurpose;
  }

  // Pre-fill targetUsers if extracted
  if (e?.targetUsers && !r.targetUsers) {
    r.targetUsers = e.targetUsers;
  }

  // IMAGE — smart pre-fills
  if (session.appType === 'image') {

    // Detect transformation intent
    const transformWords = ['convert','transform',
      'turn into','make into','change into',
      'into superhero','into anime','into cartoon',
      'into oil painting','into sketch'];
    if (!r.imageTransformationType &&
        transformWords.some(w =>
          purpose.includes(w) || msg.includes(w))) {
      r.imageTransformationType =
        'Transform person into a character/style';
    }

    // Detect style from keywords
    if (!r.imageStyle) {
      if (purpose.includes('superhero') ||
          purpose.includes('marvel') ||
          purpose.includes('comic')) {
        r.imageStyle = 'Comic book / Marvel style';
        if (!r.imageMood) r.imageMood = 'Powerful and heroic';
      }
      if (purpose.includes('anime') ||
          purpose.includes('manga')) {
        r.imageStyle = 'Anime / manga style';
      }
      if (purpose.includes('realistic') ||
          purpose.includes('photo')) {
        r.imageStyle = 'Photorealistic with enhancements';
      }
      if (purpose.includes('cartoon')) {
        r.imageStyle = 'Cartoon / illustrated';
      }
    }

    // Detect input type from keywords
    if (!r.imageInputType) {
      if (purpose.includes('convert image') ||
          purpose.includes('upload') ||
          purpose.includes('my photo') ||
          purpose.includes('their photo') ||
          purpose.includes('from image') ||
          purpose.includes('from photo')) {
        r.imageInputType = 'Upload a photo of themselves';
      }
      if (purpose.includes('text') &&
          !purpose.includes('image')) {
        r.imageInputType = 'Type a text description only';
      }
    }
  }

  // VIDEO — smart pre-fills
  if (session.appType === 'video') {
    if (!r.videoType) {
      if (purpose.includes('animate') ||
          purpose.includes('photo to video') ||
          purpose.includes('image to video')) {
        r.videoType = 'Animate a still photo into video';
        if (!r.imageInputType) r.imageInputType = 'Upload a reference image';
      }
    }
    if (!r.videoEffects) {
      if (purpose.includes('cinematic') ||
          purpose.includes('camera')) {
        r.videoEffects = 'Smooth cinematic camera movement';
      }
    }
    if (!r.videoDuration) {
      if (purpose.includes('reel') ||
          purpose.includes('short') ||
          purpose.includes('social')) {
        r.videoDuration = '5-10 seconds (medium)';
      }
    }
  }

  // TEXT — smart pre-fills
  if (session.appType === 'text') {
    if (!r.textOutputType) {
      if (purpose.includes('blog')) {
        r.textOutputType = 'Blog posts and articles';
      }
      if (purpose.includes('email')) {
        r.textOutputType = 'Email and newsletters';
      }
      if (purpose.includes('caption') ||
          purpose.includes('social media')) {
        r.textOutputType = 'Social media captions';
      }
      if (purpose.includes('product')) {
        r.textOutputType = 'Product descriptions';
      }
      if (purpose.includes('script')) {
        r.textOutputType = 'Scripts and screenplays';
      }
    }
  }

  session.requirements = r;
  return session;
}

// ═══════════════════════════════════════════════════
// CONVERSATIONAL ACKNOWLEDGMENT BUILDER
// Acknowledges the previous answer naturally before
// asking the next question
// ═══════════════════════════════════════════════════

export function buildAcknowledgment(prevField, prevAnswer, nextQuestion) {
  const acks = {
    imageTransformationType: `Got it — ${prevAnswer}.`,
    imageStyle: `${prevAnswer} — nice choice.`,
    imageMood: `${prevAnswer} works well for this.`,
    imageInputType: `Understood.`,
    imageOutputDetails: `Perfect.`,
    videoType: `Got it.`,
    videoEffects: `Good.`,
    videoDuration: `Noted.`,
    videoAudio: `Got it.`,
    targetUsers: `That helps.`,
    primaryUseCase: `Makes sense.`,
    textOutputType: `Got it.`,
    textTone: `Good.`,
    textLength: `Noted.`,
    textInputType: `Understood.`,
    audioType: `Understood.`,
    audioVoiceStyle: `Good.`,
    audioDuration: `Noted.`,
    visionTask: `Got it.`,
    visionOutputFormat: `Understood.`,
    visionAccuracy: `Noted.`,
    teamSize: `Got it.`,
    integrationNeeds: `Understood.`,
    volumeExpectation: `Noted.`
  };

  const ack = acks[prevField] || 'Got it.';

  return `${ack} ${nextQuestion}`;
}
