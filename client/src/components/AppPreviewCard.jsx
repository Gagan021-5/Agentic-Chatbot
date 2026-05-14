import React, { useState, useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import { testPreview } from '../utils/api';
import {
  livePreviewStorageKey,
  loadLivePreviewFromStorage,
  saveLivePreviewToStorage
} from '../utils/livePreviewStorage';

export default function AppPreviewCard({ data, onSendMessage, sessionId, storageMessageId }) {
  const [isEditing, setIsEditing] = useState(false);
  const [editInstruction, setEditInstruction] = useState('');
  const [isPreviewMode, setIsPreviewMode] = useState(false);
  const [testInputs, setTestInputs] = useState(() => {
    const rawVars = data?.variables || data?.variablesUsed || [];
    return rawVars.reduce((acc, variable) => {
      // Handle both object format and old string format
      const varName = typeof variable === "object" ? variable.name : String(variable || "").replace(/^\$\$/, "");

      // THE FIX: 1. Try what the user already typed. 2. Try AI test value. 3. Fallback to empty.
      const testVal =
        typeof variable === "object" && variable.value
          ? variable.value
          : variable.test_value || "";

      acc[varName] = testVal;
      return acc;
    }, {});
  });
  const [previewResult, setPreviewResult] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [imageError, setImageError] = useState(false);
  const [testImage, setTestImage] = useState(null);

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

  if (!data) return null;

  // Format variables beautifully
  const formatPrompt = (text) => {
    if (!text) return null;
    const parts = text.split(/(\$\$\w+)/g);
    return parts.map((part, i) =>
      part.startsWith('$$') ? <span key={i} className="text-rent-purple font-semibold">{part}</span> : part
    );
  };

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
    <div className="w-full bg-[#121018] border border-[#2a2238] rounded-2xl overflow-hidden mt-2 shadow-xl font-sans animate-fade-in-up">

      {/* ─── Header ─── */}
      <div className="bg-[#121018] p-5 border-b border-[#2a2238] flex justify-between items-start">
        <div>
          <h3 className="text-xl font-bold text-white mb-1">{data.appName || 'Your AI App'}</h3>
          <p className="text-sm text-gray-400">{data.appDescription || 'Ready to deploy.'}</p>
        </div>
        <div className="bg-[#2d1b4e] border border-[#5a32a3] px-3 py-1 rounded-full flex items-center shrink-0 ml-2">
          <span className="text-[#a77bf3] text-sm font-medium">{data.cost || '0.00'} coins / run</span>
        </div>
      </div>

      {/* ─── Body ─── */}
      <div className="p-5">
        {/* Default view: show prompts */}
        {!isPreviewMode && !isEditing && (
          <div className="space-y-4">
            <div>
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Backend Logic</span>
              <div className="mt-1 text-sm text-gray-300 bg-[#0a0a0f] p-3 rounded-lg border border-[#2a2238] font-mono leading-relaxed h-24 overflow-y-auto custom-scrollbar">
                {data.systemPrompt}
              </div>
            </div>
            {data.userPrompt && (
              <div>
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Prompt Template</span>
                <div className="mt-1 text-sm text-gray-300 bg-[#0a0a0f] p-3 rounded-lg border border-[#2a2238] font-mono leading-relaxed custom-scrollbar">
                  {formatPrompt(data.userPrompt)}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── Live Preview Mode ─── */}
        {isPreviewMode && (
          <div className="space-y-4 bg-[#0a0a0f] p-4 rounded-xl border border-[#3b2d50]">
            <div className="flex items-center gap-2">
              <div className="h-2 w-2 rounded-full bg-[#7c3aed] animate-pulse" />
              <h4 className="text-sm font-semibold text-white">Live Preview — Test Your App</h4>
            </div>
            <p className="text-xs text-gray-400 mb-3">
              Enter sample data below to see how your <span className="font-bold text-[#a77bf3] uppercase">{data.appType || 'TEXT'}</span> app will respond.
            </p>

            {rawVariables.map((v, i) => {
              const varName = typeof v === 'object' ? v.name : String(v || "").replace(/^\$\$/, "");
              const placeholderText = v.placeholder || `Enter ${varName}...`;
              
              return (
                <div key={i} className="flex flex-col gap-1.5">
                  <label className="text-xs text-gray-300 ml-1 font-medium">{varName}</label>
                  <input
                    type="text"
                    value={testInputs[varName] || ''}
                    onChange={(e) => handleTestInputChange(varName, e.target.value)}
                    placeholder={placeholderText}
                    className="w-full bg-[#121018] border border-[#2a2238] rounded-lg px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-[#8b5cf6]/60 transition-all placeholder:text-gray-600"
                  />
                </div>
              );
            })}

            {data.appType?.toLowerCase() === 'vision' && (
              <div className="flex flex-col gap-1.5 mb-3">
                <label className="text-xs text-[#a77bf3] font-semibold uppercase tracking-wider ml-1">Upload Test Image</label>
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if(file) {
                      const reader = new FileReader();
                      reader.onloadend = () => setTestImage(reader.result);
                      reader.readAsDataURL(file);
                    }
                  }}
                  className="text-xs text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-[#2a2238] file:text-[#a77bf3] hover:file:bg-[#3b2d50]"
                />
              </div>
            )}

            <button
              onClick={handleRunTest}
              disabled={isGenerating}
              className="w-full mt-2 py-2.5 bg-[#7c3aed] hover:bg-[#6d28d9] disabled:bg-[#3b2d50] disabled:text-gray-500 text-white rounded-lg text-sm font-semibold transition-all flex items-center justify-center gap-2 shadow-lg shadow-[#7c3aed]/20 disabled:shadow-none"
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

                {/* Audio Output (ElevenLabs MP3 from server) */}
                {previewResult.type === 'audio' && (
                  <div className="bg-[#1a1525] rounded-xl border border-[#3b2d50] p-5 mb-3 shadow-lg">
                    <div className="mb-4">
                      <h4 className="text-sm font-bold text-gray-200">Audio preview</h4>
                      <p className="text-xs text-gray-400 font-medium mt-0.5">
                        {previewResult.url
                          ? "Generated with ElevenLabs (voice from ELEVENLABS_VOICE_ID on the server)."
                          : "Run Live Test again to regenerate audio."}
                      </p>
                    </div>
                    {previewResult.url && (
                      <audio
                        controls
                        className="w-full h-10 rounded-md bg-[#121018] mb-4"
                        key={previewResult.url.slice(0, 120)}
                      >
                        <source src={previewResult.url} type="audio/mpeg" />
                      </audio>
                    )}
                    <div className="p-4 bg-black/40 rounded-lg text-sm text-gray-300 leading-relaxed border border-white/5">
                      <span className="text-xs font-semibold text-[#a77bf3] uppercase tracking-wide block mb-2">
                        Script (spoken)
                      </span>
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
      <div className="p-4 bg-[#0a0a0f] border-t border-[#2a2238]">
        {!isEditing ? (
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={() => onSendMessage('Publish App')}
              className="px-5 py-2.5 bg-[#6d28d9] hover:bg-[#5b21b6] text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-[#6d28d9]/20"
            >
              Publish to Marketplace
            </button>
            <button
              onClick={() => setIsPreviewMode(!isPreviewMode)}
              className={`px-4 py-2.5 rounded-lg text-sm font-medium transition-colors border ${
                isPreviewMode
                  ? 'bg-[#2a2238] text-white border-[#3b2d50]'
                  : 'bg-transparent text-[#a77bf3] border-[#5a32a3] hover:bg-[#2a2238]'
              }`}
            >
              {isPreviewMode ? '✕ Close Preview' : '⚡ Live Preview'}
            </button>
            <button
              onClick={() => onSendMessage('Save Draft')}
              className="px-4 py-2.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-sm font-medium transition-colors"
            >
              Save Draft
            </button>
            <button
              onClick={() => setIsEditing(true)}
              className="px-4 py-2 bg-[#2a2a2a] text-gray-300 hover:text-white rounded-lg text-sm font-medium ml-auto transition-colors"
            >
              Edit
            </button>
          </div>
        ) : (
          <div className="flex gap-2">
            <input
              type="text"
              value={editInstruction}
              onChange={(e) => setEditInstruction(e.target.value)}
              placeholder="Tell the AI what to change..."
              className="flex-1 bg-black/50 border border-[#2a2238] rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-[#8b5cf6] transition-colors"
              onKeyDown={(e) => e.key === 'Enter' && handleEditSubmit()}
            />
            <button
              onClick={handleEditSubmit}
              className="px-4 py-2 bg-[#6d28d9] text-white rounded-lg text-sm font-medium hover:bg-[#5b21b6] transition-colors"
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
