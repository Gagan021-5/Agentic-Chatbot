const MODELS = {
  text: [
    { id: "minimax-m2.7", name: "MiniMax M2.7", cost: 1.12, tier: "fast", tags: ["multimodal", "chat", "efficient"], desc: "Fast and scalable multimodal model" },
    { id: "kimi-k2-thinking", name: "Kimi K2 Thinking", cost: 3.50, tier: "premium", tags: ["reasoning", "analysis", "deep-thinking"], desc: "Advanced reasoning and structured problem solving" },
    { id: "gpt-5.2", name: "GPT-5.2", cost: 2.00, tier: "balanced", tags: ["coding", "agentic", "reasoning"], desc: "Flagship model for coding and tasks" },
    { id: "gpt-5.1", name: "GPT-5.1", cost: 8.00, tier: "premium", tags: ["deep-reasoning", "long-context"], desc: "High accuracy reasoning and planning" },
    { id: "grok-4", name: "Grok 4", cost: 20.40, tier: "ultra", tags: ["reasoning", "large-context"], desc: "xAI advanced reasoning model" },
    { id: "o4-mini", name: "O4 Mini", cost: 6.00, tier: "balanced", tags: ["reasoning", "efficient"], desc: "Fast and efficient reasoning model" },
    { id: "gpt-4.1-nano", name: "GPT-4.1 Nano", cost: 0.63, tier: "fast", tags: ["cheap", "fast"], desc: "Ultra low latency model" },
    { id: "gpt-4.1-mini", name: "GPT-4.1 Mini", cost: 1.69, tier: "balanced", tags: ["fast", "efficient"], desc: "Balanced speed and intelligence" },
    { id: "o3-mini", name: "O3 Mini", cost: 3.29, tier: "balanced", tags: ["math", "science", "reasoning"], desc: "Strong reasoning for technical tasks" },
    { id: "gpt-4o", name: "GPT-4o", cost: 1.09, tier: "premium", tags: ["multimodal", "chat", "vision"], desc: "Versatile flagship multimodal model" },
    { id: "gpt-4o-mini", name: "GPT-4o Mini", cost: 0.50, tier: "fast", tags: ["cheap", "chat"], desc: "Compact and cost-effective model" },
    { id: "llama3.3-70b", name: "LLaMA 3.3 70B", cost: 0.35, tier: "fast", tags: ["open-source", "cheap"], desc: "Affordable open-source LLM" }
  ],

  image: [
    { id: "vgpt-image-2", name: "vGPT Image 2", cost: 18.00, tier: "premium", tags: ["editing", "detailed"], desc: "Advanced image generation with editing" },
    { id: "kling-image-v3", name: "Kling Image v3", cost: 3.00, tier: "balanced", tags: ["creative", "detailed"], desc: "High-quality visual generation" },
    { id: "nano-banana-pro-edit", name: "Nano Banana Pro Edit", cost: 14.00, tier: "premium", tags: ["editing"], desc: "Fast image editing model" },
    { id: "nano-banana-2", name: "Nano Banana 2", cost: 8.50, tier: "balanced", tags: ["fast", "google"], desc: "Fast image generation/editing" },
    { id: "seedream-5-lite", name: "Seedream 5 Lite", cost: 5.15, tier: "balanced", tags: ["multi-image", "design"], desc: "Creative image generation with reasoning" },
    { id: "recraft-v4-pro", name: "Recraft v4 Pro", cost: 4.60, tier: "balanced", tags: ["design", "text-rendering"], desc: "Design-focused image model" },
    { id: "flux-2-pro", name: "Flux 2 Pro", cost: 2.50, tier: "fast", tags: ["realistic", "editing"], desc: "High-quality and efficient model" },
    { id: "imagen-4", name: "Imagen 4", cost: 3.69, tier: "premium", tags: ["photorealistic"], desc: "Google flagship image model" },
    { id: "sdxl", name: "SDXL", cost: 1.00, tier: "fast", tags: ["open-source"], desc: "Stable diffusion advanced model" },
    { id: "flux-schnell", name: "Flux Schnell", cost: 0.50, tier: "fast", tags: ["ultra-fast"], desc: "Fastest image generation model" }
  ],

  audio: [
    { id: "lyria-3-pro", name: "Lyria 3 Pro", cost: 8.50, tier: "premium", tags: ["music", "creative"], desc: "High-quality music generation" },
    { id: "tts-1.5-max", name: "TTS 1.5 Max", cost: 5.70, tier: "premium", tags: ["voice", "realistic"], desc: "Human-like speech synthesis" },
    { id: "orpheus-tts", name: "Orpheus TTS", cost: 4.49, tier: "balanced", tags: ["tts", "expressive"], desc: "Emotionally expressive speech" },
    { id: "kokoro-82m", name: "Kokoro 82M", cost: 0.79, tier: "fast", tags: ["multilingual", "cheap"], desc: "Efficient multilingual TTS" },
    { id: "stable-audio", name: "Stable Audio Open", cost: 2.00, tier: "fast", tags: ["sound", "music"], desc: "Audio generation from prompts" }
  ],

  video: [
    { id: "seedance-2.0", name: "Seedance 2.0", cost: 169.00, tier: "premium", tags: ["image-to-video", "bytedance"], desc: "Advanced cinematic animation", supports_image_input: true },
    { id: "gen-4.5", name: "Gen 4.5", cost: 56.40, tier: "premium", tags: ["runway", "cinematic"], desc: "Next-gen video model" },
    { id: "kling-v2.6-motion-control", name: "Kling v2.6 Motion", cost: 59.20, tier: "premium", tags: ["motion-control"], desc: "Precise motion video generation" },
    { id: "seedance-1.5-pro", name: "Seedance 1.5 Pro", cost: 24.00, tier: "balanced", tags: ["multimodal"], desc: "Balanced video model" },
    { id: "grok-imagine", name: "Grok Imagine", cost: 24.00, tier: "balanced", tags: ["creative"], desc: "Short AI videos with audio" },
    { id: "pixverse-v5.6", name: "PixVerse v5.6", cost: 35.00, tier: "balanced", tags: ["short", "creative"], desc: "Stylized video clips" },
    { id: "ray-2-720p", name: "Ray 2", cost: 77.50, tier: "premium", tags: ["realistic"], desc: "Luma video model" },
    { id: "wan-2.2-fast", name: "Wan 2.2 Fast", cost: 6.30, tier: "fast", tags: ["cheap"], desc: "Fast text-to-video" },
    { id: "veo-3-fast", name: "Veo 3 Fast", cost: 273.00, tier: "ultra", tags: ["google", "audio"], desc: "Fast Google Veo model" },
    { id: "veo3", name: "Veo 3", cost: 318.29, tier: "ultra", tags: ["best", "cinematic"], desc: "Most advanced video AI" }
  ]
};

export default MODELS;