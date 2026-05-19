import { useState } from "react";

const APP_TYPE_META = {
  text:   { label: "Text AI",    color: "from-blue-500/20 to-blue-600/10",   border: "border-blue-500/30",   text: "text-blue-400",   icon: "✦" },
  image:  { label: "Image AI",   color: "from-purple-500/20 to-purple-600/10", border: "border-purple-500/30", text: "text-purple-400", icon: "🖼" },
  audio:  { label: "Audio AI",   color: "from-pink-500/20 to-pink-600/10",   border: "border-pink-500/30",   text: "text-pink-400",   icon: "🎙" },
  video:  { label: "Video AI",   color: "from-orange-500/20 to-orange-600/10", border: "border-orange-500/30", text: "text-orange-400", icon: "🎬" },
  vision: { label: "Vision AI",  color: "from-teal-500/20 to-teal-600/10",   border: "border-teal-500/30",   text: "text-teal-400",   icon: "👁" },
};

function EditableField({ label, value, multiline = false, onChange }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);

  function handleBlur() {
    setEditing(false);
    onChange(draft);
  }

  return (
    <div className="group relative rounded-xl border border-white/[0.06] bg-white/[0.03] p-3.5 transition-colors hover:border-white/10">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">{label}</span>
        <button
          type="button"
          onClick={() => { setDraft(value); setEditing(true); }}
          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-lg hover:bg-white/10 text-gray-500 hover:text-gray-300"
          title={`Edit ${label}`}
        >
          <svg viewBox="0 0 16 16" fill="none" className="h-3 w-3">
            <path d="m3.5 11.8.9-2.8L10.2 3.3a1.1 1.1 0 0 1 1.6 0l.8.8a1.1 1.1 0 0 1 0 1.6L6.8 11.5l-2.8.9-.5-.6Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
          </svg>
        </button>
      </div>

      {editing ? (
        multiline ? (
          <textarea
            autoFocus
            rows={3}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={handleBlur}
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:ring-1 focus:ring-purple-500/50 resize-none"
          />
        ) : (
          <input
            autoFocus
            type="text"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleBlur()}
            className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-3 py-1.5 text-sm text-gray-200 outline-none focus:ring-1 focus:ring-purple-500/50"
          />
        )
      ) : (
        <p className="text-sm text-gray-200 leading-relaxed">{value || <span className="text-gray-600 italic">Not set</span>}</p>
      )}
    </div>
  );
}

export default function SEOPreviewCard({ data, onSendMessage }) {
  const typeMeta = APP_TYPE_META[data.appType?.toLowerCase()] || APP_TYPE_META.text;

  const [appName, setAppName]               = useState(data.appName || "");
  const [appDescription, setAppDescription] = useState(data.appDescription || "");
  const [tagInput, setTagInput]             = useState((data.tags || []).join(", "));
  const [editingTags, setEditingTags]       = useState(false);

  const parsedTags = tagInput.split(",").map(t => t.trim()).filter(Boolean).slice(0, 10);

  function handleConfirm(action) {
    const payload = {
      appName,
      appDescription,
      category: data.category,
      tags: parsedTags,
      appType: data.appType,
      modelId: data.modelId,
      costPerRun: data.costPerRun
    };
    // Send ONE message — prefix is parsed by stepRouter, never shown raw in chat
    if (action === 'publish') {
      onSendMessage(`SEO_PUBLISH::${JSON.stringify(payload)}`);
    } else if (action === 'draft') {
      onSendMessage(`SEO_DRAFT::${JSON.stringify(payload)}`);
    } else {
      onSendMessage(`SEO_EDIT::${JSON.stringify(payload)}`);
    }
  }

  return (
    <div className="w-full rounded-2xl overflow-hidden border border-white/[0.06] bg-[#0d0b12] shadow-2xl animate-fade-in-up">

      {/* Header */}
      <div className={`bg-gradient-to-r ${typeMeta.color} border-b ${typeMeta.border} p-5`}>
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-lg">{typeMeta.icon}</span>
              <span className={`text-xs font-bold uppercase tracking-widest ${typeMeta.text}`}>{typeMeta.label}</span>
              {data.category && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-white/10 text-gray-400 border border-white/10">
                  {data.category}
                </span>
              )}
            </div>
            <h2 className="text-xl font-extrabold text-white leading-tight">{appName || "Your App"}</h2>
          </div>

          {/* Cost badge */}
          <div className="shrink-0 text-right">
            <div className="text-[10px] text-gray-500 uppercase tracking-widest mb-0.5">Cost per run</div>
            <div className="text-lg font-extrabold text-white">
              {data.costPerRun ?? "—"}
              <span className="text-xs font-normal text-gray-400 ml-1">coins</span>
            </div>
          </div>
        </div>
      </div>

      {/* Editable fields */}
      <div className="p-4 space-y-3">
        <EditableField label="App Name" value={appName} onChange={setAppName} />
        <EditableField label="Description" value={appDescription} multiline onChange={setAppDescription} />

        {/* Tags */}
        <div className="group relative rounded-xl border border-white/[0.06] bg-white/[0.03] p-3.5 transition-colors hover:border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-gray-500">Tags</span>
            <button
              type="button"
              onClick={() => setEditingTags(!editingTags)}
              className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded-lg hover:bg-white/10 text-gray-500 hover:text-gray-300"
            >
              <svg viewBox="0 0 16 16" fill="none" className="h-3 w-3">
                <path d="m3.5 11.8.9-2.8L10.2 3.3a1.1 1.1 0 0 1 1.6 0l.8.8a1.1 1.1 0 0 1 0 1.6L6.8 11.5l-2.8.9-.5-.6Z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
          {editingTags ? (
            <textarea
              autoFocus
              rows={2}
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onBlur={() => setEditingTags(false)}
              placeholder="tag1, tag2, tag3..."
              className="w-full bg-black/30 border border-purple-500/30 rounded-lg px-3 py-2 text-sm text-gray-200 outline-none focus:ring-1 focus:ring-purple-500/50 resize-none"
            />
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {parsedTags.length > 0 ? parsedTags.map(tag => (
                <span key={tag} className="text-xs px-2.5 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-300 font-medium">
                  #{tag}
                </span>
              )) : <span className="text-gray-600 italic text-sm">No tags</span>}
            </div>
          )}
        </div>

        {/* Model info */}
        {data.modelId && (
          <div className="flex items-center gap-2 px-3.5 py-2.5 rounded-xl bg-white/[0.02] border border-white/[0.05]">
            <span className="text-[10px] text-gray-500 uppercase tracking-widest">🤖 AI Model</span>
            <span className="text-xs font-semibold text-purple-300 ml-auto">{data.modelName || data.modelId}</span>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="px-4 pb-4 flex flex-wrap gap-2">
        <button
          onClick={() => handleConfirm('publish')}
          className="flex-1 py-3 bg-[#7c3aed] hover:bg-[#6d28d9] text-white rounded-xl text-sm font-bold transition-all duration-200 shadow-lg shadow-purple-900/30 hover:shadow-[0_0_20px_rgba(124,58,237,0.35)] active:scale-95"
        >
          🚀 Publish to Marketplace
        </button>
        <button
          onClick={() => handleConfirm('draft')}
          className="px-4 py-3 bg-white/[0.05] hover:bg-white/10 text-gray-300 hover:text-white rounded-xl text-sm font-medium border border-white/[0.08] transition-all duration-200 active:scale-95"
        >
          Save Draft
        </button>
        <button
          onClick={() => onSendMessage('Edit App')}
          className="px-4 py-3 bg-transparent text-gray-500 hover:text-gray-300 rounded-xl text-sm font-medium transition-colors"
        >
          Edit App
        </button>
      </div>
    </div>
  );
}
