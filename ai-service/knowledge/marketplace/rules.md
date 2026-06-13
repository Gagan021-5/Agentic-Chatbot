# RentPrompts Internal Rules — Marketplace

## Content Policy
- No NSFW content
- No jailbreak or bypass prompts
- No weapons, explosives, or illegal activity guidance
- No medical diagnosis (only general health information with disclaimers)
- No legal advice presented as authoritative (must include disclaimers)
- No impersonation of real individuals

## App Quality Standards
- System prompts must define a clear AI persona
- User prompts must be in first person ("I want...")
- Variables must be user-friendly (no technical jargon)
- acceptImageInput must be correctly set based on app purpose
- Apps must produce meaningful output, not generic responses

## Pricing Guidelines
- Free tier: 0 coins — open-source models only
- Low tier: < 5 coins — fast models, simple tasks
- Medium tier: 5-20 coins — balanced quality and cost
- Premium tier: > 20 coins — best quality, complex tasks
- Ultra tier: > 100 coins — state-of-the-art models

## App Type Definitions
- TEXT: Output is written content (blogs, emails, plans, reports, recipes)
- IMAGE: Output is a picture (logos, cards, posters, room designs)
- AUDIO: Output is sound (TTS, voiceover, music, podcasts)
- VIDEO: Output is video (animations, reels, cinematic clips)
- VISION: Input is an image to ANALYZE (OCR, detection, classification)

## Variable Extraction Rules
- Detect [variable] or $$variable patterns
- Generate structured variable definitions with:
  - identifier: snake_case machine name
  - displayName: human-readable label
  - type: string, number, boolean, enum, image_url
  - placeholder: helpful example value
- Never expose internal system parameters as variables
- Maximum 8 variables per app
- Minimum 2 variables per app

## acceptImageInput Rules
- TRUE only when app TRANSFORMS or ANALYZES existing images
  - Background removal, room redesign, face swap, photo editing
  - Plant disease detection, OCR, quality inspection
- FALSE when app CREATES from text descriptions
  - Logo designer, poster creator, avatar generator
  - Any app where user DESCRIBES what they want
