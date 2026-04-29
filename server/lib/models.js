const MODELS = {
  text: [
    { id: "minimax-m2.7", name: "MiniMax M2.7", cost: 20.40, tier: "balanced", tags: ["text", "creative", "no-settings"], desc: "Advanced text, simple setup" },
    { id: "gpt-4o", name: "GPT-4o", cost: 15.00, tier: "premium", tags: ["reasoning", "chat", "best-quality"], desc: "Best for reasoning and chat" },
    { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", cost: 4.00, tier: "fast", tags: ["fast", "cheap", "multimodal"], desc: "Fast cheap multimodal text" },
    { id: "llama-3.3-70b", name: "LLaMA 3.3 70B", cost: 2.00, tier: "fast", tags: ["cheap", "open-source", "fast"], desc: "Open source, very affordable" }
  ],
  image: [
    { id: "flux-kontext-max", name: "Flux Kontext Max", cost: 8.00, tier: "balanced", tags: ["professional", "realistic", "editing"], desc: "Professional image generation" },
    { id: "kling-image-v3", name: "Kling Image v3", cost: 12.00, tier: "premium", tags: ["high-quality", "creative", "detailed"], desc: "High quality detailed images" },
    { id: "gemini-flash-image", name: "Gemini Flash Image", cost: 4.00, tier: "fast", tags: ["fast", "cheap", "simple"], desc: "Fast affordable image generation" }
  ],
  audio: [
    { id: "elevenlabs-v2", name: "ElevenLabs v2", cost: 10.00, tier: "premium", tags: ["voice", "realistic", "tts"], desc: "Most realistic voice synthesis" },
    { id: "suno-v4", name: "Suno v4", cost: 15.00, tier: "premium", tags: ["music", "songs", "creative"], desc: "AI music and song generation" },
    { id: "whisper-v3", name: "Whisper v3", cost: 3.00, tier: "fast", tags: ["transcription", "stt", "fast"], desc: "Speech to text transcription" }
  ],
  video: [
    { id: "seedance-2.0", name: "Seedance 2.0", cost: 0, tier: "free", tags: ["free", "image-to-video", "bytedance"], desc: "FREE — ByteDance image to video", supports_image_input: true },
    { id: "wan-2.2-fast", name: "Wan 2.2 T2V Fast", cost: 6.30, tier: "fast", tags: ["cheap", "fast", "text-to-video"], desc: "Very fast cheap text-to-video" },
    { id: "seedance-1.5-pro", name: "Seedance 1.5 Pro", cost: 24.00, tier: "balanced", tags: ["balanced", "multimodal", "5s-10s"], desc: "Advanced multimodal video" },
    { id: "grok-imagine", name: "Grok Imagine", cost: 24.00, tier: "balanced", tags: ["balanced", "creative"], desc: "Grok video generation" },
    { id: "pixverse-v5.6", name: "PixVerse v5.6", cost: 35.00, tier: "balanced", tags: ["short", "creative", "stylized"], desc: "Short creative video clips" },
    { id: "gen-4-turbo", name: "Gen 4 Turbo", cost: 38.29, tier: "balanced", tags: ["controllable", "flexible", "runway"], desc: "Controllable flexible video" },
    { id: "gen-4.5", name: "Gen 4.5", cost: 55.50, tier: "premium", tags: ["premium", "runway", "next-gen"], desc: "Runway next generation video" },
    { id: "kling-v2.6-motion", name: "Kling v2.6 Motion Control", cost: 60.00, tier: "premium", tags: ["motion-control", "cinematic", "image-to-video"], desc: "Advanced cinematic motion control", supports_image_input: true },
    { id: "kling-v2.6", name: "Kling v2.6", cost: 63.00, tier: "premium", tags: ["cinematic", "fluid", "audio"], desc: "Cinematic fluid motion video" },
    { id: "seedance-1-pro", name: "Seedance 1 Pro", cost: 68.25, tier: "premium", tags: ["text-to-video", "image-to-video", "5s-10s"], desc: "Text and image to video" },
    { id: "ray-2-720p", name: "Ray 2 720p", cost: 77.50, tier: "premium", tags: ["large-scale", "luma", "720p"], desc: "Luma multi-modal large model" },
    { id: "veo-3-fast", name: "Veo 3 Fast", cost: 273.00, tier: "ultra", tags: ["google", "ultra", "with-sound"], desc: "Google Veo 3 faster version" },
    { id: "veo3", name: "Veo 3", cost: 318.29, tier: "ultra", tags: ["google", "best", "with-sound", "ultra"], desc: "Most advanced video AI + sound" }
  ],
  vision: [
    { id: "gpt-4o-vision", name: "GPT-4o Vision", cost: 15.00, tier: "premium", tags: ["analysis", "understanding", "best"], desc: "Best image understanding" },
    { id: "gemini-vision", name: "Gemini Flash Vision", cost: 5.00, tier: "fast", tags: ["fast", "cheap", "analysis"], desc: "Fast image analysis" },
    { id: "claude-vision", name: "Claude Vision", cost: 12.00, tier: "premium", tags: ["detailed", "reasoning", "visual"], desc: "Detailed visual reasoning" }
  ]
};

export default MODELS;
