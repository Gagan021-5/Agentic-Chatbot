const MODELS = {
  text: [
    { id: "minimax-m2.7", name: "MiniMax M2.7", cost: 20.4, tier: "balanced", desc: "Advanced text, no extra settings" },
    { id: "gpt-4o", name: "GPT-4o", cost: 15.0, tier: "premium", desc: "Best for reasoning and chat" },
    { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", cost: 4.0, tier: "fast", desc: "Fast cheap multimodal text" },
    { id: "llama-3.3-70b", name: "LLaMA 3.3 70B", cost: 2.0, tier: "fast", desc: "Open source, very cheap" }
  ],
  image: [
    { id: "flux-kontext-max", name: "Flux Kontext Max", cost: 8.0, tier: "balanced", desc: "Professional image generation" },
    { id: "kling-image-v3", name: "Kling Image v3", cost: 12.0, tier: "premium", desc: "High quality images" },
    { id: "gemini-flash-image", name: "Gemini Flash Image", cost: 4.0, tier: "fast", desc: "Fast image generation" }
  ],
  audio: [
    { id: "elevenlabs-v2", name: "ElevenLabs v2", cost: 10.0, tier: "premium", desc: "Best voice synthesis" },
    { id: "suno-v4", name: "Suno v4", cost: 15.0, tier: "premium", desc: "AI music generation" },
    { id: "whisper-v3", name: "Whisper v3", cost: 3.0, tier: "fast", desc: "Speech to text transcription" }
  ],
  video: [
    { id: "seedance-2.0", name: "Seedance 2.0", cost: 0, tier: "free", desc: "ByteDance i2v, FREE", supports_image_input: true },
    { id: "wan-2.2-fast", name: "Wan 2.2 T2V Fast", cost: 6.3, tier: "fast", desc: "Very fast cheap text-to-video" },
    { id: "seedance-1.5-pro", name: "Seedance 1.5 Pro", cost: 24.0, tier: "balanced", desc: "Advanced multimodal video" },
    { id: "grok-imagine", name: "Grok Imagine", cost: 24.0, tier: "balanced", desc: "Grok video generation" },
    { id: "pixverse-v5.6", name: "PixVerse v5.6", cost: 35.0, tier: "balanced", desc: "Short creative videos" },
    { id: "gen-4-turbo", name: "Gen 4 Turbo", cost: 38.29, tier: "balanced", desc: "Controllable flexible video" },
    { id: "gen-4.5", name: "Gen 4.5", cost: 55.5, tier: "premium", desc: "Runway next-gen video" },
    { id: "kling-v2.6-motion", name: "Kling v2.6 Motion Control", cost: 60.0, tier: "premium", desc: "Advanced motion control", supports_image_input: true },
    { id: "kling-v2.6", name: "Kling v2.6", cost: 63.0, tier: "premium", desc: "Cinematic fluid motion video" },
    { id: "seedance-1-pro", name: "Seedance 1 Pro", cost: 68.25, tier: "premium", desc: "Text and image to video 5s/10s" },
    { id: "ray-2-720p", name: "Ray 2 720p", cost: 77.5, tier: "premium", desc: "Luma multi-modal large model" },
    { id: "veo-3-fast", name: "Veo 3 Fast", cost: 273.0, tier: "ultra", desc: "Google Veo 3 faster version" },
    { id: "veo3", name: "Veo 3", cost: 318.29, tier: "ultra", desc: "Most advanced video AI, with sound" }
  ],
  vision: [
    { id: "gpt-4o-vision", name: "GPT-4o Vision", cost: 15.0, tier: "premium", desc: "Best image understanding + analysis" },
    { id: "gemini-vision", name: "Gemini Flash Vision", cost: 5.0, tier: "fast", desc: "Fast image analysis and description" },
    { id: "claude-vision", name: "Claude Vision", cost: 12.0, tier: "premium", desc: "Detailed visual reasoning" }
  ]
};

export default MODELS;
