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
  const isFree = model.cost === 0;

  return (
    <div className="card-hover group relative flex h-full flex-col overflow-hidden rounded-2xl border border-rent-border glass-panel p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      {/* Top accent glow on hover */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-rent-purple/0 to-transparent transition-all duration-300 group-hover:via-rent-purple/40" />

      <div className="flex items-start justify-between gap-2 sm:gap-3">
        <div className="min-w-0">
          <h3 className="text-base font-extrabold text-white sm:text-lg">{model.name}</h3>
          <p className="mt-1.5 text-xs leading-5 text-white/50 sm:mt-2 sm:text-sm sm:leading-6">{model.desc}</p>
        </div>
        <div className={`flex shrink-0 items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold sm:px-3 sm:py-1.5 sm:text-xs ${
          isFree
            ? "border-green-500/20 bg-green-500/8 text-green-300"
            : "border-yellow-500/20 bg-yellow-500/8 text-yellow-300"
        }`}>
          {isFree ? "🎉" : <CoinIcon />}
          {formatCost(model.cost)}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-1.5 sm:mt-4 sm:gap-2">
        <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-[0.14em] sm:px-3 sm:py-1 sm:text-xs ${tierStyles[model.tier] || tierStyles.fast}`}>
          {isUltra ? <WarningIcon /> : null}
          {model.tier}
        </span>
        {model.supports_image_input ? (
          <span className="rounded-full border border-sky-500/15 bg-sky-500/8 px-2.5 py-0.5 text-[10px] font-medium text-sky-300/70 sm:px-3 sm:py-1 sm:text-xs">
            🖼️ image input
          </span>
        ) : null}
      </div>

      {/* Tags */}
      {model.tags && model.tags.length > 0 ? (
        <div className="mt-3 flex flex-wrap gap-1 sm:mt-4 sm:gap-1.5">
          {model.tags.slice(0, 4).map((tag) => (
            <span key={tag} className="rounded-md border border-white/5 bg-white/[0.03] px-2 py-0.5 text-[9px] font-medium text-white/30 sm:text-[10px]">
              {tag}
            </span>
          ))}
        </div>
      ) : null}

      <div className="mt-auto pt-4 sm:pt-5">
        <button
          type="button"
          onClick={() => onSendMessage(`Select ${model.id}`)}
          className="btn-cta h-10 w-full rounded-xl text-xs font-bold text-white sm:h-12 sm:text-sm"
        >
          Select Model
        </button>
      </div>
    </div>
  );
}

export default ModelCard;
