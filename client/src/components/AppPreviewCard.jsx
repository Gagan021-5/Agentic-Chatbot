import React, { useState, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import { testPreview } from '../utils/api';
import {
  livePreviewStorageKey,
  loadLivePreviewFromStorage,
  saveLivePreviewToStorage
} from '../utils/livePreviewStorage';

export default function AppPreviewCard({ data, onSendMessage, sessionId, storageMessageId }) {
  const [isEditing, setIsEditing] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [editInstruction, setEditInstruction] = useState('');
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [testInputs, setTestInputs] = useState(() => {
    const rawVars = data?.variables || data?.variablesUsed || [];
    return rawVars.reduce((acc, variable) => {
      const varName = typeof variable === "object" ? variable.name : String(variable || "").replace(/^\$\$/, "");

      // FIX: Start entirely empty! Let the user type their own case details.
      acc[varName] = "";
      return acc;
    }, {});
  });
  const [previewResult, setPreviewResult] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [imageError, setImageError] = useState(false);
  const [testImage, setTestImage] = useState(null);
  const [testImageInfo, setTestImageInfo] = useState(null); // { w, h, kb }

  const storageKey = useMemo(
    () => livePreviewStorageKey(sessionId, storageMessageId),
    [sessionId, storageMessageId]
  );

  useLayoutEffect(() => {
    if (!storageKey) return;
    const s = loadLivePreviewFromStorage(storageKey);
    if (!s) return;
    if (typeof s.isPreviewMode === "boolean") setIsPreviewMode(s.isPreviewMode);
    if (s.testInputs && typeof s.testInputs === "object") {
      setTestInputs((prev) => ({ ...prev, ...s.testInputs }));
    }
    if (s.previewResult) setPreviewResult(s.previewResult);
    if (s.testImage) setTestImage(s.testImage);
  }, [storageKey]);

  const prevStorageKeyRef = useRef(null);
  const skipNextPersistRef = useRef(true);

  useEffect(() => {
    if (!storageKey) return;
    if (prevStorageKeyRef.current !== storageKey) {
      prevStorageKeyRef.current = storageKey;
      skipNextPersistRef.current = true;
    }
    if (skipNextPersistRef.current) {
      skipNextPersistRef.current = false;
      return;
    }
    const t = setTimeout(() => {
      saveLivePreviewToStorage(storageKey, {
        isPreviewMode,
        testInputs,
        previewResult,
        testImage
      });
    }, 450);
    return () => clearTimeout(t);
  }, [storageKey, isPreviewMode, testInputs, previewResult, testImage]);

  useEffect(() => {
    if (previewResult?.url) setImageError(false);
  }, [previewResult?.url]);

  /** Resize + compress an image File to a base64 JPEG, max 900px on longest side. */
  function compressImageToBase64(file, maxPx = 900, quality = 0.85) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const { width, height } = img;
          const scale = Math.min(1, maxPx / Math.max(width, height));
          const w = Math.round(width * scale);
          const h = Math.round(height * scale);
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          const dataUrl = canvas.toDataURL('image/jpeg', quality);
          const kb = Math.round((dataUrl.length * 3) / 4 / 1024);
          resolve({ dataUrl, w, h, kb });
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  if (!data) return null;

  // Format variables beautifully
  const formatPrompt = (text) => {
    if (!text) return null;
    const parts = text.split(/(\$\$\w+)/g);
    return parts.map((part, i) =>
      part.startsWith('$$') ? <span key={i} className="text-rent-purple font-semibold">{part}</span> : part
    );
  };

  // Convert snake_case or camelCase variable names to readable Title Case labels
  const toLabel = (name) =>
    String(name || '')
      .replace(/([a-z])([A-Z])/g, '$1 $2')   // camelCase → words
      .replace(/[_-]+/g, ' ')                  // snake_case → words
      .replace(/\b\w/g, (c) => c.toUpperCase()) // capitalize each word
      .trim();

  const handleEditSubmit = () => {
    if (editInstruction.trim()) {
      onSendMessage(`Change: ${editInstruction}`);
      setIsEditing(false);
      setEditInstruction('');
    }
  };

  // Support both data.variables (new) and data.variablesUsed (existing)
  const rawVariables = data.variables || data.variablesUsed || [];
  const normalizedVariables = rawVariables.map((v) => {
    if (typeof v === 'object' && v.name) return v.name;
    return String(v || "").replace(/^\$\$/, "");
  }).filter(Boolean);

  const handleTestInputChange = (name, value) => {
    setTestInputs(prev => ({ ...prev, [name]: value }));
  };

  const handleRunTest = async () => {
    setIsGenerating(true);
    setPreviewResult(null);
    setPreviewError('');
    setImageError(false);
    try {
      // 1. Intelligent Fallback: If appType is missing, guess it from the prompt
      let currentAppType = data.appType;
      if (!currentAppType) {
        const promptText = (data.systemPrompt || '').toLowerCase();
        if (promptText.includes('image') || promptText.includes('photo') || promptText.includes('realistic')) {
          currentAppType = 'image';
        } else if (promptText.includes('audio') || promptText.includes('voice')) {
          currentAppType = 'audio';
        } else {
          currentAppType = 'text';
        }
      }

      const json = await testPreview({
        appType: currentAppType,
        variables: testInputs,
        systemPrompt: data.systemPrompt,
        testImageBase64: testImage
      });
      if (json.success) {
        setPreviewResult(json.preview);
      } else {
        setPreviewError(json.error || "Preview generation failed.");
      }
    } catch (e) {
      console.error(e);
      setPreviewError(e.message || "Network error.");
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="w-full bg-[#121018] border border-white/[0.06] rounded-2xl overflow-hidden mt-2 shadow-xl font-sans animate-fade-in-up">

      {/* ─── Body ─── */}
      <div className="p-6">
        {/* Default view: show prompts */}
        {!isPreviewMode && !isEditing && (
          <div className="space-y-4">
            <div>
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">System Prompt</span>
              <div className="mt-1.5 text-sm text-gray-300 bg-[#0a0a0f] p-4 rounded-xl border border-white/[0.05] font-mono leading-relaxed h-24 overflow-y-auto custom-scrollbar">
                {data.systemPrompt}
              </div>
            </div>
            {data.userPrompt && (
              <div>
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">User Prompt</span>
                <div className="mt-1.5 text-sm text-gray-300 bg-[#0a0a0f] p-4 rounded-xl border border-white/[0.05] font-mono leading-relaxed custom-scrollbar">
                  {formatPrompt(data.userPrompt)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── Live Preview Mode ─── */}
        {isPreviewMode && (
          <div className="space-y-4 bg-[#0a0a0f] p-5 rounded-xl border border-purple-500/15">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-[#7c3aed] animate-pulse" />
              <h4 className="text-sm font-semibold text-white">Live Preview — Test Your App</h4>
            </div>
            <p className="text-xs text-gray-400 mb-3">
              Enter sample data below to see how your <span className="font-bold text-[#a77bf3] uppercase">{data.appType || 'TEXT'}</span> app will respond.
            </p>

            {rawVariables.map((v, i) => {
              const varName = typeof v === 'object' ? v.name : String(v || "").replace(/^\$\$/, "");

              // FIX: Use the AI's test_value as a helpful placeholder hint
              const placeholderText = typeof v === 'object' && v.test_value
                ? `e.g., ${v.test_value}`
                : (typeof v === 'object' && v.placeholder ? v.placeholder : `Enter ${varName}...`);

              return (
                <div key={i} className="flex flex-col gap-1.5">
                  <label className="text-xs text-gray-300 ml-1 font-medium">{toLabel(varName)}</label>
                  <textarea
                    rows={1}
                    value={testInputs[varName] || ''}
                    onChange={(e) => handleTestInputChange(varName, e.target.value)}
                    onInput={(e) => {
                      e.target.style.height = 'auto';
                      e.target.style.height = e.target.scrollHeight + 'px';
                    }}
                    placeholder={placeholderText}
                    className="w-full bg-[#121018] border border-white/[0.06] rounded-xl px-4 py-2.5 text-sm text-gray-200 outline-none focus:outline-none focus:ring-1 focus:ring-purple-500/50 focus:border-purple-500/30 transition-all duration-200 placeholder:text-gray-600 resize-none overflow-hidden leading-relaxed"
                  />
                </div>
              );
            })}

            {/* Voice gender selector for audio apps */}
            {data.appType?.toLowerCase() === 'audio' && (
              <div className="flex flex-col gap-2 p-3 rounded-xl border border-purple-500/20 bg-purple-500/5">
                <label className="text-xs text-[#a77bf3] font-semibold uppercase tracking-wider">🎙 Voice Gender</label>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => handleTestInputChange('voice_gender', 'female')}
                    className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-all ${
                      testInputs['voice_gender'] !== 'male'
                        ? 'bg-pink-500/20 border-pink-500/40 text-pink-300'
                        : 'bg-white/5 border-white/10 text-gray-500 hover:border-white/20'
                    }`}
                  >
                    ♀ Female (Natalie)
                  </button>
                  <button
                    type="button"
                    onClick={() => handleTestInputChange('voice_gender', 'male')}
                    className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-all ${
                      testInputs['voice_gender'] === 'male'
                        ? 'bg-blue-500/20 border-blue-500/40 text-blue-300'
                        : 'bg-white/5 border-white/10 text-gray-500 hover:border-white/20'
                    }`}
                  >
                    ♂ Male (Terrell)
                  </button>
                </div>
              </div>
            )}

            {/* Image upload — shown ONLY when the app needs user to provide a source image.
                acceptImageInput is set by the LLM based on app purpose:
                  TRUE:  background removal, room redesign, portrait editing, style transfer
                  FALSE: logo generator, poster creator, text-to-image art generator */}
            {data.acceptImageInput && (
              <div className="flex flex-col gap-1.5 mb-3">
                <label className="text-xs text-[#a77bf3] font-semibold uppercase tracking-wider ml-1">
                  {data.appType?.toLowerCase() === 'vision'
                    ? '📷 Upload Image to Analyze'
                    : '🖼 Upload Your Source Image'}
                </label>
                <p className="text-xs text-gray-500 ml-1 -mt-0.5">
                  {data.appType?.toLowerCase() === 'vision'
                    ? 'Upload the image this app will analyze'
                    : 'Upload the photo you want to transform'}
                </p>
                {testImage && (
                  <div className="relative w-full rounded-xl overflow-hidden border border-purple-500/20 mb-2">
                    <img src={testImage} alt="Preview" className="w-full max-h-48 object-contain bg-black/40" />
                    <button
                      onClick={() => setTestImage(null)}
                      className="absolute top-2 right-2 bg-black/70 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-500/80 transition-colors"
                    >✕</button>
                  </div>
                )}
                <input
                  type="file"
                  accept="image/*"
                  onChange={async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;
                    try {
                      const { dataUrl, w, h, kb } = await compressImageToBase64(file);
                      setTestImage(dataUrl);
                      setTestImageInfo({ w, h, kb });
                    } catch {
                      // Fallback: send raw (server limit is now 15MB)
                      const reader = new FileReader();
                      reader.onloadend = () => setTestImage(reader.result);
                      reader.readAsDataURL(file);
                    }
                  }}
                  className="text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-[#2a2238] file:text-[#a77bf3] hover:file:bg-[#3b2d50] cursor-pointer"
                />
              </div>
            )}


            <button
              onClick={handleRunTest}
              disabled={isGenerating}
              className="w-full mt-3 py-3 bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#3b2d50] disabled:text-gray-500 text-white rounded-xl text-sm font-semibold transition-all duration-300 ease-in-out flex items-center justify-center gap-2 shadow-lg shadow-[#7c3aed]/20 hover:shadow-[0_0_20px_rgba(124,58,237,0.3)] disabled:shadow-none active:scale-95"
            >
              {isGenerating ? (
                <>
                  <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
                  </svg>
                  <span>Generating Preview...</span>
                </>
              ) : '▶ Run Live Test'}
            </button>

            {/* Error display */}
            {previewError && (
              <div className="mt-3 p-3 bg-red-900/20 border border-red-800/40 rounded-lg text-red-300 text-xs">
                ⚠ {previewError}
              </div>
            )}

            {/* PREVIEW RESULTS DISPLAY */}
            {previewResult && (
              <div className="mt-4 pt-4 border-t border-[#2a2238] animate-fade-in-up">
                <span className="text-xs font-semibold text-[#a77bf3] uppercase tracking-wider mb-3 block">Result</span>

                {(previewResult.type === 'image' || previewResult.type === 'multimodal') &&
                  !previewResult.url &&
                  previewResult._imageDroppedForStorage && (
                    <div className="mb-3 rounded-lg border border-amber-500/25 bg-amber-950/25 px-3 py-2.5 text-xs text-amber-100/95 leading-relaxed">
                      The image was not kept in browser storage (size limit). Fields and any text below were restored — click{" "}
                      <strong className="text-amber-50">Run Live Test</strong> to regenerate the image.
                    </div>
                  )}

                {/* Image or Multimodal Output */}
                {(previewResult.type === 'image' || previewResult.type === 'multimodal') && previewResult.url && (
                  !imageError ? (
                    <div className="mb-3 max-h-[min(85vh,960px)] overflow-auto rounded-xl border border-[#3b2d50] bg-[#0a0a0f] shadow-lg flex justify-center items-start p-1">
                      <img
                        key={previewResult.url.slice(0, 120)}
                        src={previewResult.url}
                        alt="AI Generation Preview"
                        className="max-w-full w-full h-auto object-contain rounded-lg"
                        referrerPolicy="no-referrer"
                        decoding="async"
                        onError={() => {
                          const u = previewResult.url;
                          const hint = u.startsWith("data:") ? "Invalid image data in preview." : "Failed to load image URL.";
                          console.error(hint, u.slice(0, 200));
                          setImageError(true);
                        }}
                      />
                    </div>
                  ) : (
                    <div className="w-full h-48 bg-[#1a1525] rounded-xl border border-red-500/30 flex items-center justify-center mb-3 px-4 text-center">
                      <p className="text-sm text-red-400">
                        Preview image could not be displayed. Try Run Live Test again, or check the server log if this persists.
                      </p>
                    </div>
                  )
                )}

                {/* Audio Output — ElevenLabs TTS (with Web Speech fallback) */}
                {previewResult.type === 'audio' && (
                  <div className="bg-[#1a1525] rounded-xl border border-[#3b2d50] p-5 mb-3 shadow-lg">
                    <div className="flex items-center justify-between mb-4">
                      <div>
                        <h4 className="text-sm font-bold text-gray-200">Audio Preview</h4>
                        <p className="text-xs text-gray-500 mt-0.5">
                          {previewResult.url
                            ? `Powered by Murf AI · ${previewResult.voiceLabel || 'AI Voice'}`
                            : "Powered by your browser's built-in voice engine"}
                        </p>
                      </div>

                      {/* Fallback: Web Speech play button — only shown when no ElevenLabs audio */}
                      {!previewResult.url && (
                        <button
                          onClick={() => {
                            if (!previewResult.data) return;
                            if (isSpeaking) {
                              window.speechSynthesis.cancel();
                              setIsSpeaking(false);
                              return;
                            }
                            const utter = new SpeechSynthesisUtterance(previewResult.data);
                            utter.rate = 1;
                            utter.pitch = 1;
                            utter.lang = 'en-US';
                            utter.onstart = () => setIsSpeaking(true);
                            utter.onend = () => setIsSpeaking(false);
                            utter.onerror = () => setIsSpeaking(false);
                            window.speechSynthesis.cancel();
                            window.speechSynthesis.speak(utter);
                          }}
                          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-300 ease-in-out ${
                            isSpeaking
                              ? 'bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30'
                              : 'bg-[#7c3aed] text-white hover:bg-[#6d28d9] hover:shadow-[0_0_16px_rgba(124,58,237,0.3)]'
                          }`}
                        >
                          {isSpeaking ? (
                            <>
                              <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500" />
                              </span>
                              Stop
                            </>
                          ) : (
                            <>
                              <svg viewBox="0 0 20 20" fill="currentColor" className="h-4 w-4"><path d="M6.3 2.841A1.5 1.5 0 0 0 4 4.11V15.89a1.5 1.5 0 0 0 2.3 1.269l9.344-5.89a1.5 1.5 0 0 0 0-2.538L6.3 2.84z"/></svg>
                              Play
                            </>
                          )}
                        </button>
                      )}
                    </div>

                    {/* ElevenLabs native audio player */}
                    {previewResult.url && (
                      <audio
                        src={previewResult.url}
                        controls
                        autoPlay
                        className="w-full rounded-lg mb-4 accent-[#7c3aed]"
                        style={{ colorScheme: 'dark' }}
                      />
                    )}

                    <div className="p-4 bg-black/40 rounded-lg text-sm text-gray-300 leading-relaxed border border-white/5">
                      <span className="text-xs font-semibold text-[#a77bf3] uppercase tracking-wide block mb-2">Script (spoken)</span>
                      <p className="whitespace-pre-wrap">{previewResult.data}</p>
                    </div>
                  </div>
                )}

                {/* Video Output */}
                {previewResult.type === 'video' && (
                  <div className="mb-3 animate-fade-in">
                    {/* Fake Video Player Frame */}
                    <div className="relative w-full aspect-video rounded-t-xl overflow-hidden shadow-lg border border-[#3b2d50] border-b-0 group cursor-pointer bg-black">
                      <img
                        src={previewResult.url}
                        alt="Video Thumbnail"
                        className="w-full h-full object-cover opacity-70 group-hover:opacity-90 transition-opacity duration-500"
                        referrerPolicy="no-referrer"
                        onError={(e) => {
                          e.target.src =
                            "https://via.placeholder.com/1024x576/1a1525/a77bf3?text=Video+Thumbnail";
                        }}
                      />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-16 h-16 bg-black/60 backdrop-blur-md border border-white/20 rounded-full flex items-center justify-center text-white text-2xl shadow-2xl pl-1 group-hover:scale-110 transition-transform">
                          ▶
                        </div>
                      </div>
                      <div className="absolute top-3 left-3 bg-red-500 text-[10px] font-bold px-2 py-0.5 rounded text-white tracking-wider animate-pulse">
                        LIVE
                      </div>
                    </div>
                    {/* Screenplay text block */}
                    <div className="p-4 bg-[#1a1525] border border-[#3b2d50] rounded-b-xl text-sm text-gray-300 whitespace-pre-wrap leading-relaxed shadow-lg">
                      <h4 className="text-[#a77bf3] font-bold mb-3 uppercase tracking-wide text-xs">Video Concept & Screenplay</h4>
                      {previewResult.data}
                    </div>
                  </div>
                )}

                {/* Text or Multimodal Output (Text Portion) */}
                {(previewResult.type === 'text' || previewResult.type === 'multimodal') && previewResult.content && (
                  <div className="bg-[#1a1525] p-3 rounded-lg border border-[#2a2238]">
                    <p className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed">{previewResult.content}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ─── Actions Footer ─── */}
      <div className="p-5 bg-[#0a0a0f] border-t border-white/[0.06]">
        {!isEditing ? (
          <div className="flex flex-wrap gap-2 items-center">
            {/* Live Preview toggle always present */}
            <button
              onClick={() => setIsPreviewMode(!isPreviewMode)}
              className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-300 ease-in-out border ${
                isPreviewMode
                  ? 'bg-[#2a2238] text-white border-[#3b2d50]'
                  : 'bg-transparent text-[#a77bf3] border-[#5a32a3] hover:bg-[#2a2238]'
              }`}
            >
              {isPreviewMode ? '✕ Close Preview' : '⚡ Live Preview'}
            </button>

            {/* Action buttons driven by data.options — only what the backend sends */}
            {(data.options || []).map((option) => {
              const isPublish = option.toLowerCase().includes('publish');
              const isApprove = option.toLowerCase().includes('approve');
              const isDraft   = option.toLowerCase().includes('draft') || option.toLowerCase().includes('save');
              const isEdit    = option.toLowerCase().includes('edit');

              if (isPublish) return (
                <button key={option}
                  onClick={() => onSendMessage('Publish App')}
                  className="px-5 py-2.5 bg-[#6d28d9] hover:bg-[#5b21b6] text-white rounded-xl text-sm font-medium transition-all duration-300 ease-in-out shadow-lg shadow-[#6d28d9]/20 hover:shadow-[0_0_20px_rgba(109,40,217,0.3)] active:scale-95"
                >{option}</button>
              );
              if (isApprove) return (
                <button key={option}
                  onClick={() => onSendMessage('Approve App')}
                  className="px-5 py-2.5 bg-[#6d28d9] hover:bg-[#5b21b6] text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-[#6d28d9]/20"
                >
                  Approve & Continue
                </button>
              );
              if (isDraft) return (
                <button key={option}
                  onClick={() => onSendMessage('Save Draft')}
                  className="px-4 py-2.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-sm font-medium transition-colors"
                >{option}</button>
              );
              if (isEdit) return (
                <button key={option}
                  onClick={() => setIsEditing(true)}
                  className="px-4 py-2 bg-[#2a2a2a] text-gray-300 hover:text-white rounded-xl text-sm font-medium ml-auto transition-all duration-300 ease-in-out hover:bg-[#333] active:scale-95"
                >{option}</button>
              );
              // Generic fallback button
              return (
                <button key={option}
                  onClick={() => onSendMessage(option)}
                  className="px-4 py-2.5 bg-transparent text-gray-300 hover:text-white rounded-lg text-sm font-medium border border-white/10 transition-colors"
                >{option}</button>
              );
            })}
          </div>
        ) : (
          <div className="flex gap-2">
            <input
              type="text"
              value={editInstruction}
              onChange={(e) => setEditInstruction(e.target.value)}
              placeholder="Tell the AI what to change..."
              className="flex-1 bg-black/50 border border-white/[0.06] rounded-xl px-4 py-2.5 text-sm text-white outline-none focus:outline-none focus:ring-1 focus:ring-purple-500/50 focus:border-purple-500/30 transition-all duration-300"
              onKeyDown={(e) => e.key === 'Enter' && handleEditSubmit()}
            />
            <button
              onClick={handleEditSubmit}
              className="px-4 py-2.5 bg-[#6d28d9] text-white rounded-xl text-sm font-medium hover:bg-[#5b21b6] transition-all duration-300 ease-in-out active:scale-95"
            >
              Update
            </button>
            <button
              onClick={() => setIsEditing(false)}
              className="px-3 py-2 text-white/50 hover:text-white text-sm transition-colors"
            >
              Cancel
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
