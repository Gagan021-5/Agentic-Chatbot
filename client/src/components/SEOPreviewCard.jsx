import { useState } from "react";

function PencilButton({ onClick }) {
  return (
    <button type="button" onClick={onClick} className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-rent-border bg-white/4 text-white/45 transition hover:bg-white/8 hover:text-white sm:h-8 sm:w-8 sm:rounded-xl">
      <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5">
        <path d="m4.5 14.8 1.1-3.4L12.7 4.3a1.4 1.4 0 0 1 2 0l1 1a1.4 1.4 0 0 1 0 2l-7.1 7.1-3.4 1.1Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      </svg>
    </button>
  );
}

function SEOPreviewCard({ data, onSendMessage }) {
  const [draft, setDraft] = useState({
    appName: data.appName || "",
    appDescription: data.appDescription || "",
    tags: Array.isArray(data.tags) ? data.tags : [],
    category: data.category || "",
    suggestedPrice: data.suggestedPrice
  });
  const [editingField, setEditingField] = useState(null);
  const [tagInput, setTagInput] = useState((data.tags || []).join(", "));

  function confirmChanges() {
    const payload = {
      ...draft,
      tags: tagInput.split(",").map((t) => t.trim()).filter(Boolean).slice(0, 10)
    };
    onSendMessage(`Confirm SEO::${JSON.stringify(payload)}`);
  }

  return (
    <div className="rounded-2xl border border-rent-border bg-rent-card p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/35 sm:text-xs">SEO Preview</div>
          <div className="mt-1.5 text-xl font-extrabold text-white sm:mt-2 sm:text-2xl">
            {editingField === "appName" ? (
              <input autoFocus value={draft.appName} onBlur={() => setEditingField(null)} onChange={(e) => setDraft((c) => ({ ...c, appName: e.target.value }))} className="w-full rounded-xl border border-rent-border bg-rent-surface px-3 py-2 text-lg text-white outline-none sm:text-xl" />
            ) : draft.appName}
          </div>
        </div>
        <PencilButton onClick={() => setEditingField("appName")} />
      </div>

      <div className="mt-4 rounded-xl border border-rent-border bg-black/15 p-3 sm:mt-5 sm:rounded-2xl sm:p-4">
        <div className="flex items-start justify-between gap-2 sm:gap-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/35 sm:text-xs">Description</div>
          <PencilButton onClick={() => setEditingField("appDescription")} />
        </div>
        <div className="mt-2 text-xs leading-6 text-white/70 sm:mt-3 sm:text-sm sm:leading-7">
          {editingField === "appDescription" ? (
            <textarea autoFocus rows={4} value={draft.appDescription} onBlur={() => setEditingField(null)} onChange={(e) => setDraft((c) => ({ ...c, appDescription: e.target.value }))} className="w-full rounded-xl border border-rent-border bg-rent-surface px-3 py-2.5 text-sm text-white outline-none" />
          ) : draft.appDescription}
        </div>
      </div>

      <div className="mt-3 rounded-xl border border-rent-border bg-black/15 p-3 sm:mt-4 sm:rounded-2xl sm:p-4">
        <div className="flex items-start justify-between gap-2 sm:gap-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/35 sm:text-xs">Tags</div>
          <PencilButton onClick={() => setEditingField("tags")} />
        </div>
        {editingField === "tags" ? (
          <textarea autoFocus rows={3} value={tagInput} onBlur={() => setEditingField(null)} onChange={(e) => setTagInput(e.target.value)} className="mt-2 w-full rounded-xl border border-rent-border bg-rent-surface px-3 py-2.5 text-sm text-white outline-none sm:mt-3" />
        ) : (
          <div className="mt-2 flex flex-wrap gap-1.5 sm:mt-3 sm:gap-2">
            {tagInput.split(",").map((t) => t.trim()).filter(Boolean).slice(0, 10).map((tag) => (
              <span key={tag} className="rounded-full border border-rent-border bg-rent-elevated px-2.5 py-1 text-[10px] font-semibold text-white/65 sm:px-3 sm:py-1.5 sm:text-xs">{tag}</span>
            ))}
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2 text-xs text-white/45 sm:mt-4 sm:gap-3 sm:text-sm">
        <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-1 sm:px-3 sm:py-1.5">Category: {draft.category}</span>
        <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-1 sm:px-3 sm:py-1.5">Suggested: {draft.suggestedPrice} coins</span>
      </div>

      <button type="button" onClick={confirmChanges} className="btn-cta mt-4 h-10 w-full rounded-xl text-xs font-bold text-white sm:mt-5 sm:h-12 sm:w-auto sm:px-6 sm:text-sm">
        Confirm & Continue
      </button>
    </div>
  );
}

export default SEOPreviewCard;
