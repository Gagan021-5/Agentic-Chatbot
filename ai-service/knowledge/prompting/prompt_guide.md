# Prompt Engineering Guide — RentPrompts Best Practices

## General Principles

### System Prompt Structure
1. Define AI persona in second person: "You are..."
2. Include: role, domain expertise, tone, output format, constraints
3. Keep to 3-5 sentences for most apps
4. Never mention model names, costs, or platform internals

### User Prompt Structure (Text/Audio)
1. Write in first person: "I want...", "My topic is..."
2. Include scenario with $$variables
3. Add step-by-step processing logic
4. Specify output format (headings, bullets, length)
5. Add constraints (what NOT to do)
6. Target 200-400 words

### User Prompt Structure (Image/Video)
1. Write as single flowing visual description
2. Incorporate $$variables naturally
3. Include art direction: lighting, camera angle, colors, mood
4. End with quality keywords
5. Keep 50-120 words — image models don't read long prompts
6. NO markdown headers or numbered steps

## Variable Design Rules

### User Perspective Principle
Variables must reflect what a NON-EXPERT end-user can provide:
- WRONG: section_number, diagnosis_code, statute_citation
- RIGHT: incident_description, symptoms, situation_type
- The AI should DERIVE technical answers from plain descriptions

### Naming Conventions
- Use snake_case for identifiers
- Display names should be human-readable: "Company Name" not "company_name"
- Avoid generic names: input, text, data, main_input
- Use domain-specific names: incident_details, dispute_context, target_aesthetic

### Variable Types
- string: Most common — free text input
- number: Counts, ages, quantities
- boolean: Toggles and flags
- enum: Predefined choices (style, tone, language)
- image_url: When app needs image upload

## Domain-Specific Patterns

### Legal Apps
- User describes incident → AI identifies applicable sections
- Include cross-validation instructions for contradictory inputs
- Never hallucinate section numbers or case outcomes
- Variables: incident_description, jurisdiction, desired_outcome

### Medical/Health Apps
- User describes symptoms → AI provides guidance
- Always recommend professional consultation
- Variables: symptoms, medical_history, lifestyle_factors

### Creative/Design Apps
- Focus on visual description language
- Include style, mood, color palette, composition
- Variables: subject, style, color_scheme, composition

### Education Apps
- Structure output pedagogically
- Include difficulty level and learning objectives
- Variables: topic, difficulty_level, student_age, learning_format

## Anti-Patterns to Avoid
1. Never use "You want..." in user prompts (use "I want...")
2. Never mention model names in prompts
3. Never use markdown in image/video prompts
4. Never ask users for technical/expert-level inputs
5. Never generate generic "assistant" prompts
6. Never include coin costs or platform metadata
