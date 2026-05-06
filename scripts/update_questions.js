const fs = require('fs');
const path = require('path');
const file = path.join(__dirname, '..', 'server', 'lib', 'stepRouter.js');
let content = fs.readFileSync(file, 'utf8');

const newDQ = `const getDeepQuestions = (appType) => ({
  image: [
    { field: 'imageStyle', question: \`Understood. For this \${appType} generator, what visual style should the output have?\`, options: ['Photorealistic / photography','Comic book / superhero style','Anime / manga','Oil painting / artistic','Cinematic / dramatic','Cartoon / illustrated','I want to choose the style'] },
    { field: 'imageInputType', question: \`Got it. What will you provide to generate the \${appType}?\`, options: ['Just a text description','Upload a photo of myself','Upload any reference image','Both text and photo upload'] },
    { field: 'imageUseCase', question: \`What will you mainly use this \${appType} tool for?\`, options: ['Social media profile pictures','Fun personal use','Marketing and branding','Gaming avatars','Gifts and merchandise','Professional headshots','Something else'] }
  ],
  video: [
    { field: 'videoType', question: \`Got it. What kind of \${appType} should this app create?\`, options: ['Animate a still photo into video','Text description to video','Cinematic scenes','Short social media reels','Product showcase videos','Talking avatar / presenter'] },
    { field: 'videoEffects', question: \`Understood. What motion or visual effect does the \${appType} need?\`, options: ['Smooth cinematic camera movement','Dynamic action sequences','Slow motion dramatic effect','Natural realistic motion','I will choose the effect'] },
    { field: 'videoDuration', question: \`Got it. For this \${appType} tool, what duration do you want to set?\`, options: ['3-5 seconds','5-10 seconds','10-30 seconds','I will set the duration'] }
  ],
  text: [
    { field: 'textPurpose', question: \`Alright. What exactly should this \${appType} app generate or plan?\`, options: ['Workout / fitness plans','Meal / diet plans','Blog posts and articles','Social media captions','Email and newsletters','Product descriptions','Study / learning plans','Travel itineraries','Scripts and screenplays','Something else'] },
    { field: 'textTone', question: \`Got it. What tone should the generated \${appType} have?\`, options: ['Professional and formal','Casual and friendly','Motivational and energetic','Educational and clear','Creative and expressive','I will control the tone'] },
    { field: 'textPersonalization', question: \`Should the app personalize the \${appType} content for you?\`, options: ['Yes — based on my goals and preferences','Yes — based on my input each time','No — fixed template with variables','Not sure yet'] }
  ],
  audio: [
    { field: 'audioType', question: \`Alright. What kind of \${appType} should this app generate?\`, options: ['Voice narration / text to speech','AI music generation','Sound effects','Podcast production','Voice cloning','Speech to text transcription'] },
    { field: 'audioStyle', question: \`Got it. What voice or \${appType} style is needed?\`, options: ['Professional narrator voice','Warm conversational tone','Energetic / motivational voice','Multiple language support','I will pick from voice options'] }
  ],
  vision: [
    { field: 'visionTask', question: \`Understood. What should this \${appType} app do when it sees an image?\`, options: ['Describe what is in the image','Detect and label objects','Read text from image (OCR)','Analyze medical images','Inspect product quality','Answer questions about an image'] },
    { field: 'visionOutput', question: \`Got it. What format should the \${appType} analysis result be in?\`, options: ['Plain text description','Structured report','JSON for developers','Simple yes/no answer','User-friendly summary'] }
  ]
});`;

content = content.replace(/const DEEP_QUESTIONS = \{[\s\S]*?\n\};\n/, newDQ + '\n\n');
content = content.replace('const questions = DEEP_QUESTIONS[session.appType] || [];', 'const questions = getDeepQuestions(session.appType || "AI")[session.appType] || [];');
content = content.replace('question: `Got it. Before we configure the settings, what exactly do you want this ${session.appType || \'AI\'} app to generate or do? Describe your specific idea.`,', 'question: `Alright. What kind of ${session.appType || \'AI\'} app are you looking to create?`,');
content = content.replace("'appPurpose': 'Great idea.',", "'appPurpose': `Great. I understand the goal of your ${session.appType || 'AI'} app.`,");

fs.writeFileSync(file, content);
console.log('Script completed.');
