import { useState } from "react";

function renderPromptWithHighlights(text, variables = []) {
  if (!text) return null;
  const rawVars = variables || [];
  if (rawVars.length === 0) {
    const parts = String(text).split(/(\$\$[\w\-]+(?:\$\$)?)/g);
    return parts.map((part, i) =>
      part.startsWith("$$") ? (
        <span key={`${part}-${i}`} className="rounded-md bg-rent-purple/15 px-1.5 py-0.5 font-bold text-rent-purple">{part}</span>
      ) : (
        <span key={`${part}-${i}`}>{part}</span>
      )
    );
  }

  // Sort variables by length descending to match longer ones first
  const sortedVars = [...rawVars].sort((a, b) => b.length - a.length);

  const varPatterns = sortedVars.map(v => {
    const clean = String(v).replace(/^\$\$/, "").replace(/\$\$$/, "");
    const escaped = clean.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
    const flexible = escaped.split(/[_\s\-]+/).join('[_\\s\\-]+');
    return `\\$\\$${flexible}(?:\\$\\$)?`;
  });

  // Generic fallback pattern to match word variables
  varPatterns.push('\\$\\$[a-zA-Z0-9_\\-]+(?:\\$\\$)?');

  const regex = new RegExp(`(${varPatterns.join('|')})`, 'gi');
  const parts = String(text).split(regex);
  return parts.map((part, i) =>
    part.startsWith("$$") ? (
      <span key={`${part}-${i}`} className="rounded-md bg-rent-purple/15 px-1.5 py-0.5 font-bold text-rent-purple">{part}</span>
    ) : (
      <span key={`${part}-${i}`}>{part}</span>
    )
  );
}

function PromptPreviewCard({ data, onSendMessage }) {
  const [isEditing, setIsEditing] = useState(false);
  const [instruction, setInstruction] = useState("");

  function submitEdit() {
    if (!instruction.trim()) return;
    onSendMessage(`Edit prompt::${instruction.trim()}`);
    setInstruction("");
    setIsEditing(false);
  }

  return (
    <div className="glass-panel rounded-2xl border border-rent-border p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 sm:gap-3">
        <div>
          <h3 className="text-base font-extrabold text-white sm:text-lg">Prompt Preview</h3>
          <p className="mt-0.5 text-xs text-white/45 sm:mt-1 sm:text-sm">Variables stay editable for every future app run.</p>
        </div>
        <div className="flex flex-wrap gap-1.5 text-[10px] text-white/45 sm:gap-2 sm:text-xs">
          {data.advancedSettings?.aspectRatio ? <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-0.5 sm:px-3 sm:py-1">Aspect {data.advancedSettings.aspectRatio}</span> : null}
          {data.advancedSettings?.quality ? <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-0.5 sm:px-3 sm:py-1">Quality {data.advancedSettings.quality}</span> : null}
          {data.advancedSettings?.duration ? <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-0.5 sm:px-3 sm:py-1">Duration {data.advancedSettings.duration}</span> : null}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-rent-border bg-rent-surface/80 p-3 font-mono text-xs leading-6 text-white/80 sm:mt-5 sm:rounded-2xl sm:p-4 sm:text-sm sm:leading-7">
        {renderPromptWithHighlights(data.userPrompt, data.variablesUsed)}
      </div>

      {data.negativePrompt ? (
        <div className="mt-3 rounded-xl border border-red-500/15 bg-rent-surface/60 p-3 sm:mt-4 sm:rounded-2xl sm:p-4">
          <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-red-200/60 sm:text-xs">Negative prompt</div>
          <div className="mt-1.5 font-mono text-xs leading-6 text-red-100/65 sm:mt-2 sm:text-sm sm:leading-7">{data.negativePrompt}</div>
        </div>
      ) : null}

      <div className="mt-3 flex flex-wrap gap-1.5 sm:mt-4 sm:gap-2">
        {(data.variablesUsed || []).map((v) => (
          <span key={v} className="rounded-full border border-rent-purple/20 bg-rent-purple/8 px-2.5 py-1 text-[10px] font-semibold text-rent-purple sm:px-3 sm:py-1.5 sm:text-xs">
            $${v}
          </span>
        ))}
      </div>

      {/* Prompt explanation */}
      {data.promptExplanation ? (
        <p className="mt-3 text-xs italic leading-5 text-white/30 sm:mt-4 sm:text-sm sm:leading-6">
          💡 {data.promptExplanation}
        </p>
      ) : null}

      {isEditing ? (
        <div className="mt-4 flex flex-col gap-2.5 rounded-xl border border-rent-border bg-rent-surface/80 p-3 sm:mt-5 sm:gap-3 sm:rounded-2xl sm:p-4">
          <input
            value={instruction}
            onChange={(e) => setInstruction(e.target.value)}
            placeholder="Example: make it feel more premium and reduce motion."
            className="h-10 rounded-xl border border-rent-border bg-rent-bg px-3 text-xs text-white outline-none placeholder:text-white/30 transition focus:border-rent-purple/40 sm:h-12 sm:px-4 sm:text-sm"
          />
          <div className="flex flex-wrap gap-2 sm:gap-3">
            <button type="button" onClick={submitEdit} className="btn-cta h-9 rounded-xl px-4 text-xs font-bold text-white sm:h-11 sm:px-5 sm:text-sm">Apply edit</button>
            <button type="button" onClick={() => { setInstruction(""); setIsEditing(false); }} className="h-9 rounded-xl border border-rent-border bg-rent-elevated px-4 text-xs font-semibold text-white/50 transition hover:text-white/80 sm:h-11 sm:px-5 sm:text-sm">Cancel</button>
          </div>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap gap-2 sm:mt-5 sm:gap-3">
        <button type="button" onClick={() => onSendMessage("Looks good")} className="btn-cta h-10 rounded-xl px-4 text-xs font-bold text-white sm:h-12 sm:px-5 sm:text-sm">Looks good ✓</button>
        <button type="button" onClick={() => setIsEditing(true)} className="h-10 rounded-xl border border-rent-border bg-rent-elevated px-4 text-xs font-semibold text-white/50 transition hover:border-rent-border-light hover:text-white/80 sm:h-12 sm:px-5 sm:text-sm">Edit this</button>
      </div>
    </div>
  );
}

export default PromptPreviewCard;
