import { useState } from "react";

export default function FinalBlueprintCard({ data }) {
  const [copiedAll, setCopiedAll] = useState(false);
  const [nameCopied, setNameCopied] = useState(false);
  const [sysCopied, setSysCopied] = useState(false);
  const [userCopied, setUserCopied] = useState(false);

  if (!data) return null;

  const tags = Array.isArray(data.tags) ? data.tags : [];
  const variables = Array.isArray(data.variables) ? data.variables : [];
  const costDisplay = data.costPerRun != null ? `${data.costPerRun} coins` : "N/A";

  const handleCopyName = () => {
    navigator.clipboard.writeText(data.appName || "").then(() => {
      setNameCopied(true);
      setTimeout(() => setNameCopied(false), 1800);
    });
  };

  const handleCopySys = () => {
    navigator.clipboard.writeText(data.systemPrompt || "").then(() => {
      setSysCopied(true);
      setTimeout(() => setSysCopied(false), 1800);
    });
  };

  const handleCopyUser = () => {
    navigator.clipboard.writeText(data.userPrompt || "").then(() => {
      setUserCopied(true);
      setTimeout(() => setUserCopied(false), 1800);
    });
  };

  const handleCopyAll = () => {
    const manifest = [
      `App Name: ${data.appName || "N/A"}`,
      `Description: ${data.appDescription || "N/A"}`,
      `App Type: ${data.appType || "N/A"}`,
      `Model: ${data.modelId || "N/A"}`,
      `Cost: ${costDisplay}`,
      `Category: ${data.category || "N/A"}`,
      `Tags: ${tags.join(", ") || "N/A"}`,
      `Rapp ID: ${data.rappId || "N/A"}`,
      "",
      "--- System Prompt ---",
      data.systemPrompt || "",
      "",
      "--- User Prompt ---",
      data.userPrompt || "",
      "",
      "--- Variables ---",
      ...variables.map(v => `  ${v.name || v}: ${v.placeholder || ""}`)
    ].join("\n");

    navigator.clipboard.writeText(manifest).then(() => {
      setCopiedAll(true);
      setTimeout(() => setCopiedAll(false), 2000);
    });
  };

  return (
    <div className="w-full bg-[#0d0b14] border border-purple-500/20 rounded-2xl overflow-hidden mt-2 shadow-2xl font-sans animate-fade-in-up">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600/20 via-purple-500/10 to-transparent px-6 py-4 border-b border-purple-500/15">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500/20 text-purple-300">
              <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5">
                <path d="M9 12l2 2 4-4m5 2a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-white">{data.appName || "Your App Blueprint"}</h3>
              <p className="text-xs text-gray-400 mt-0.5">Published & Ready to Dev</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleCopyAll}
            className="inline-flex items-center gap-2 rounded-xl border border-purple-500/30 bg-purple-500/10 px-4 py-2 text-xs font-semibold text-purple-300 transition-all hover:bg-purple-500/20 hover:border-purple-500/50 active:scale-95"
          >
            {copiedAll ? "✓ Copied Manifest!" : "📋 Copy Full Manifest"}
          </button>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {/* Header Metrics Grid */}
        <div>
          <h4 className="text-[11px] font-bold text-purple-400/80 uppercase tracking-widest mb-2">Metrics Dashboard</h4>
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
            {/* App Name Micro-card */}
            <div className="relative bg-[#121018] rounded-xl border border-white/[0.05] p-4 flex flex-col justify-between overflow-hidden group hover:border-purple-500/30 transition-all duration-200">
              <div className="absolute top-2.5 right-2.5">
                <button
                  type="button"
                  onClick={handleCopyName}
                  className="text-gray-400 hover:text-purple-300 transition-all px-2 py-1 rounded-md bg-white/5 border border-white/10 hover:bg-purple-500/15"
                  title="Copy App Name"
                >
                  <span className="text-[10px] font-semibold">{nameCopied ? "Copied! ✓" : "Copy"}</span>
                </button>
              </div>
              <div className="mt-1">
                <span className="text-[9px] font-bold text-purple-400/80 uppercase tracking-widest block mb-1">App Name</span>
                <h4 className="text-xs font-bold text-white pr-10 line-clamp-2 leading-relaxed">{data.appName || "N/A"}</h4>
              </div>
            </div>

            {/* Description Micro-card */}
            <div className="bg-[#121018] rounded-xl border border-white/[0.05] p-4 flex flex-col justify-between group hover:border-purple-500/30 transition-all duration-200">
              <div>
                <span className="text-[9px] font-bold text-purple-400/80 uppercase tracking-widest block mb-1">Description</span>
                <p className="text-xs text-gray-300 line-clamp-2 leading-relaxed">{data.appDescription || "No description provided."}</p>
              </div>
            </div>

            {/* Target Model Micro-card */}
            <div className="bg-[#121018] rounded-xl border border-white/[0.05] p-4 flex flex-col justify-between group hover:border-purple-500/30 transition-all duration-200">
              <div>
                <span className="text-[9px] font-bold text-purple-400/80 uppercase tracking-widest block mb-1">Target Model ID</span>
                <code className="text-xs text-purple-300 font-mono line-clamp-2">{data.modelId || "N/A"}</code>
              </div>
            </div>

            {/* Cost Per Run Micro-card */}
            <div className="bg-[#121018] rounded-xl border border-white/[0.05] p-4 flex flex-col justify-between group hover:border-purple-500/30 transition-all duration-200">
              <div>
                <span className="text-[9px] font-bold text-purple-400/80 uppercase tracking-widest block mb-1">Cost Per Run</span>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <span className="text-sm font-extrabold text-white">{costDisplay}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tags */}
        {tags.length > 0 && (
          <div>
            <h4 className="text-[11px] font-bold text-purple-400/80 uppercase tracking-widest mb-2">Tags</h4>
            <div className="flex flex-wrap gap-1.5">
              {tags.map((tag, i) => (
                <span key={i} className="inline-block rounded-full border border-purple-500/20 bg-purple-500/10 px-3 py-1 text-[11px] font-medium text-purple-300">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* System Prompt Box */}
        {data.systemPrompt && (
          <div className="relative group border border-white/[0.05] bg-[#0a0a0f] rounded-xl overflow-hidden p-4">
            <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/[0.04]">
              <span className="text-xs font-semibold text-purple-400 uppercase tracking-wider">System Prompt</span>
              <button
                type="button"
                onClick={handleCopySys}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-gray-400 rounded-lg bg-white/5 border border-white/10 hover:bg-purple-500/15 hover:text-purple-300 hover:border-purple-500/30 transition-all active:scale-95"
              >
                {sysCopied ? (
                  <>
                    <svg viewBox="0 0 20 20" fill="none" className="h-3 w-3 text-green-400">
                      <path d="M4 10l4.5 4.5L16 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span>Copied! ✓</span>
                  </>
                ) : (
                  <>
                    <svg viewBox="0 0 20 20" fill="none" className="h-3 w-3">
                      <rect x="6" y="6" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
                      <path d="M14 6V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h1" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-h-48 overflow-y-auto custom-scrollbar">
              <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap leading-relaxed select-all">
                {data.systemPrompt}
              </pre>
            </div>
          </div>
        )}

        {/* User Prompt Box */}
        {data.userPrompt && (
          <div className="relative group border border-white/[0.05] bg-[#0a0a0f] rounded-xl overflow-hidden p-4">
            <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/[0.04]">
              <span className="text-xs font-semibold text-purple-400 uppercase tracking-wider">User Prompt</span>
              <button
                type="button"
                onClick={handleCopyUser}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-[11px] font-medium text-gray-400 rounded-lg bg-white/5 border border-white/10 hover:bg-purple-500/15 hover:text-purple-300 hover:border-purple-500/30 transition-all active:scale-95"
              >
                {userCopied ? (
                  <>
                    <svg viewBox="0 0 20 20" fill="none" className="h-3 w-3 text-green-400">
                      <path d="M4 10l4.5 4.5L16 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span>Copied! ✓</span>
                  </>
                ) : (
                  <>
                    <svg viewBox="0 0 20 20" fill="none" className="h-3 w-3">
                      <rect x="6" y="6" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
                      <path d="M14 6V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h1" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>
            <div className="max-h-48 overflow-y-auto custom-scrollbar">
              <pre className="text-xs text-gray-300 font-mono whitespace-pre-wrap leading-relaxed select-all">
                {data.userPrompt}
              </pre>
            </div>
          </div>
        )}

        {/* Variables */}
        {variables.length > 0 && (
          <div>
            <h4 className="text-[11px] font-bold text-purple-400/80 uppercase tracking-widest mb-2">Variables ({variables.length})</h4>
            <div className="grid gap-2 sm:grid-cols-2">
              {variables.map((v, i) => {
                const name = typeof v === "object" ? v.name : String(v);
                const placeholder = typeof v === "object" ? (v.placeholder || "") : "";
                return (
                  <div key={i} className="bg-[#121018] rounded-xl border border-white/[0.05] px-4 py-3">
                    <span className="text-xs font-bold text-purple-300">[{name}]</span>
                    {placeholder && <p className="text-[11px] text-gray-500 mt-0.5">{placeholder}</p>}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Pollinations.ai concept preview for image apps */}
        {data.appType?.toLowerCase() === "image" && (
          <div>
            <h4 className="text-[11px] font-bold text-purple-400/80 uppercase tracking-widest mb-2">Concept Preview</h4>
            <div className="rounded-xl border border-zinc-800 overflow-hidden bg-[#0a0a0f]">
              <img
                src={`https://image.pollinations.ai/p/${encodeURIComponent(
                  (data.userPrompt || data.systemPrompt || "beautiful AI generated artwork")
                    .split('\n')[0]
                    .replace(/\[.*?\]/g, "sample input")
                    .slice(0, 200)
                )}?width=600&height=400&nologo=true`}
                alt="AI Concept Preview"
                className="w-full h-48 object-cover rounded-xl border border-zinc-800"
                loading="lazy"
                onError={(e) => {
                  e.target.onerror = null;
                  e.target.src = "https://placehold.co/600x400/18181b/ffffff?text=Image+Preview+Ready";
                }}
              />
            </div>
            <p className="text-[10px] text-gray-600 mt-1.5 text-center">Preview powered by Pollinations.ai</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="px-6 py-3 border-t border-white/[0.04] bg-[#0a0815] flex items-center justify-between text-[10px] text-gray-600">
        <span>Rapp ID: <code className="font-mono text-purple-400">{data.rappId || "N/A"}</code></span>
        <span>🚀 Live on RentPrompts · {data.publishedAt ? new Date(data.publishedAt).toLocaleString() : "Just now"}</span>
      </div>
    </div>
  );
}
