import { useState } from "react";

function ConfirmCard({
  summary,
  detail = null,
  onYes,
  onNo,
  yesLabel = "Yes, looks good ✓",
  noLabel = "No, let me change this"
}) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [correction, setCorrection] = useState("");

  function handleNo() {
    setShowCorrection(true);
  }

  function submitCorrection() {
    if (!correction.trim()) return;
    onNo(correction.trim());
    setCorrection("");
    setShowCorrection(false);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitCorrection();
    }
  }

  return (
    <div className="mt-4 glass-panel rounded-2xl border border-rent-border p-4 shadow-soft sm:mt-5 sm:rounded-[22px] sm:p-5">
      {/* Subtle top accent line */}
      <div className="mb-3 h-px w-full bg-gradient-to-r from-transparent via-rent-purple/25 to-transparent sm:mb-4" />

      <p className="text-sm font-semibold leading-6 text-white sm:text-base sm:leading-7">
        {summary}
      </p>

      {detail ? (
        <p className="mt-1.5 text-xs leading-5 text-white/45 sm:mt-2 sm:text-sm sm:leading-6">
          {detail}
        </p>
      ) : null}

      {showCorrection ? (
        <div className="mt-3 flex flex-col gap-2.5 sm:mt-4 sm:gap-3">
          <textarea
            autoFocus
            rows={3}
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Tell me what to change..."
            className="w-full rounded-xl border border-rent-border bg-rent-surface px-3 py-2.5 text-sm text-white outline-none placeholder:text-white/30 transition focus:border-rent-purple/40 focus:shadow-input-glow sm:px-4 sm:py-3"
          />
          <div className="flex flex-wrap gap-2 sm:gap-3">
            <button
              type="button"
              onClick={submitCorrection}
              disabled={!correction.trim()}
              className="btn-cta h-9 rounded-xl px-4 text-xs font-bold text-white disabled:opacity-40 sm:h-10 sm:px-5 sm:text-sm"
            >
              Send correction
            </button>
            <button
              type="button"
              onClick={() => { setShowCorrection(false); setCorrection(""); }}
              className="h-9 rounded-xl border border-rent-border bg-rent-elevated px-4 text-xs font-semibold text-white/50 transition hover:border-rent-border-light hover:text-white/80 sm:h-10 sm:px-5 sm:text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="mt-3 flex flex-wrap gap-2 sm:mt-4 sm:gap-3">
          <button
            type="button"
            onClick={onYes}
            className="btn-cta h-9 rounded-xl px-5 text-xs font-bold text-white sm:h-10 sm:px-6 sm:text-sm"
          >
            {yesLabel}
          </button>
          <button
            type="button"
            onClick={handleNo}
            className="h-9 rounded-xl border border-rent-border bg-rent-elevated px-5 text-xs font-semibold text-white/50 transition hover:border-rent-border-light hover:text-white/80 sm:h-10 sm:px-6 sm:text-sm"
          >
            {noLabel}
          </button>
        </div>
      )}
    </div>
  );
}

export default ConfirmCard;
