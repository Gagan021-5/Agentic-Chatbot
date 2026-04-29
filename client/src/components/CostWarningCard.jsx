function formatCoins(value) {
  return Number(value || 0).toLocaleString(undefined, {
    minimumFractionDigits: value % 1 ? 2 : 0,
    maximumFractionDigits: 2
  });
}

function CostWarningCard({ data, onSendMessage }) {
  return (
    <div className="glass-panel rounded-2xl border border-red-500/20 p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-red-500/25 bg-red-500/10 text-lg text-red-300 sm:h-11 sm:w-11 sm:rounded-2xl">
          ⚠️
        </div>
        <div className="min-w-0">
          <div className="text-sm font-extrabold text-white sm:text-base">Cost check before publishing</div>
          <p className="mt-0.5 text-xs text-white/45 sm:mt-1 sm:text-sm">This model is powerful, but it can get expensive at scale.</p>
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:mt-5 sm:gap-4 md:grid-cols-2">
        <div className="rounded-xl border border-red-500/15 bg-rent-surface/80 p-3 sm:rounded-2xl sm:p-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/35 sm:text-xs">Selected model</div>
          <div className="mt-2 text-lg font-extrabold text-white sm:mt-3 sm:text-xl">{data.selectedModel}</div>
          <div className="mt-1.5 text-xs text-white/55 sm:mt-2 sm:text-sm">{formatCoins(data.selectedCost)} coins per run</div>
          <div className="mt-0.5 text-xs text-white/35 sm:mt-1 sm:text-sm">100 runs = {formatCoins(data.hundredRunCost)} coins</div>
        </div>

        <div className="rounded-xl border border-rent-border bg-rent-surface/80 p-3 sm:rounded-2xl sm:p-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/35 sm:text-xs">Cheaper alternative</div>
          <div className="mt-2 text-lg font-extrabold text-white sm:mt-3 sm:text-xl">{data.alternativeModel}</div>
          <div className="mt-1.5 text-xs text-white/55 sm:mt-2 sm:text-sm">{formatCoins(data.alternativeCost)} coins per run</div>
          <div className="mt-0.5 text-xs text-white/35 sm:mt-1 sm:text-sm">Similar output at lower burn.</div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 sm:mt-5 sm:gap-3">
        <button type="button" onClick={() => onSendMessage("Proceed anyway")} className="h-10 rounded-xl border border-red-400/20 bg-red-500/8 px-4 text-xs font-bold text-red-200 transition hover:bg-red-500/12 sm:h-12 sm:px-5 sm:text-sm">
          Proceed anyway
        </button>
        <button type="button" onClick={() => onSendMessage("Use cheaper model")} className="btn-cta h-10 rounded-xl px-4 text-xs font-bold text-white sm:h-12 sm:px-5 sm:text-sm">
          Use cheaper model
        </button>
      </div>
    </div>
  );
}

export default CostWarningCard;
