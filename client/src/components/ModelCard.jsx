const tierStyles = {
  free: "border-green-500/25 bg-green-500/10 text-green-300",
  fast: "border-sky-500/25 bg-sky-500/10 text-sky-300",
  balanced: "border-orange-500/25 bg-orange-500/10 text-orange-300",
  premium: "border-fuchsia-500/25 bg-fuchsia-500/10 text-fuchsia-300",
  ultra: "border-red-500/25 bg-red-500/10 text-red-300"
};

function CoinIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5 sm:h-4 sm:w-4">
      <circle cx="10" cy="10" r="7.2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12.6 7.6c-.5-.6-1.4-.9-2.4-.8-1.1.1-1.9.8-1.9 1.6 0 .9.9 1.3 2.2 1.6 1.4.3 2.6.7 2.6 1.9 0 .9-.8 1.7-2.2 1.8-1 .1-2-.2-2.7-.9M10 6.2v7.6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5">
      <path d="M10 3.5 16.2 15H3.8L10 3.5Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M10 7.4v3.8M10 13.4h.01" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function formatCost(cost) {
  if (cost === 0) return "FREE";
  return `${cost.toFixed(2)} coins`;
}

function ModelCard({ model, onSendMessage }) {
  const isUltra = model.tier === "ultra";

  return (
    <div className="card-hover glass-panel flex h-full flex-col rounded-2xl border border-rent-border bg-rent-card/90 p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      <div className="flex items-start justify-between gap-2 sm:gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-extrabold text-white sm:text-lg">{model.name}</h3>
          <p className="mt-1.5 text-xs leading-5 text-white/50 sm:mt-2 sm:text-sm sm:leading-6">{model.desc}</p>
        </div>
        <div className="flex shrink-0 items-center gap-1 rounded-full border border-yellow-500/20 bg-yellow-500/8 px-2.5 py-1 text-[10px] font-bold text-yellow-300 sm:px-3 sm:py-1.5 sm:text-xs">
          <CoinIcon />
          {formatCost(model.cost)}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5 sm:mt-4 sm:gap-2">
        <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.14em] sm:px-3 sm:py-1 sm:text-xs ${tierStyles[model.tier] || tierStyles.fast}`}>
          {isUltra ? <WarningIcon /> : null}
          {model.tier}
        </span>
        {model.supports_image_input ? (
          <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-0.5 text-[10px] font-medium text-white/55 sm:px-3 sm:py-1 sm:text-xs">image input</span>
        ) : null}
      </div>

      <div className="mt-4 flex-1 rounded-xl border border-white/5 bg-black/15 p-3 text-xs leading-6 text-white/45 sm:mt-5 sm:rounded-2xl sm:p-4 sm:text-sm sm:leading-7">
        Ideal for repeatable prompt workflows, clear pricing, and demo-ready marketplace positioning.
      </div>

      <button
        type="button"
        onClick={() => onSendMessage(`Select ${model.id}`)}
        className="btn-cta mt-4 h-10 w-full rounded-xl text-xs font-bold text-white sm:mt-5 sm:h-12 sm:text-sm"
      >
        Select
      </button>
    </div>
  );
}

export default ModelCard;
