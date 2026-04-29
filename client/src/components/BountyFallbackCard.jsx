function BountyFallbackCard({ data, onSendMessage }) {
  return (
    <div className="glass-panel rounded-2xl border border-amber-500/20 p-4 shadow-soft sm:rounded-[22px] sm:p-5">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-amber-500/25 bg-amber-500/10 text-lg sm:h-11 sm:w-11 sm:rounded-2xl">
          💰
        </div>
        <div className="min-w-0">
          <div className="text-sm font-extrabold text-white sm:text-base">Not enough joules right now</div>
          <p className="mt-0.5 text-xs text-white/45 sm:mt-1 sm:text-sm">
            No worries — you can post this as a bounty instead. Creators on RentPrompts will bid on your project and build it for you.
          </p>
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-amber-500/10 bg-rent-surface/80 p-3 sm:mt-5 sm:rounded-2xl sm:p-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/35 sm:text-xs">What your bounty includes</div>
        <div className="mt-2 space-y-2 text-xs text-white/65 sm:mt-3 sm:space-y-2.5 sm:text-sm">
          <div className="flex items-center gap-2">
            <span className="text-amber-400">✓</span>
            <span>App name and description</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400">✓</span>
            <span>Prompt template</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400">✓</span>
            <span>Target model: {data.modelName}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-amber-400">✓</span>
            <span>Open for creator bids</span>
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-white/40 sm:mt-4 sm:gap-3 sm:text-sm">
        <span>Your balance: {(data.userBalance || 0).toLocaleString()} joules</span>
        <span className="text-white/20">•</span>
        <span>Minimum needed: {(data.leanCost || 0).toLocaleString()} joules</span>
      </div>

      <div className="mt-4 flex flex-wrap gap-2 sm:mt-5 sm:gap-3">
        <button
          type="button"
          onClick={() => onSendMessage("Post as bounty")}
          className="btn-cta h-10 rounded-xl px-4 text-xs font-bold text-white sm:h-12 sm:px-5 sm:text-sm"
        >
          Post as Bounty
        </button>
        <a
          href="https://rentprompts.ai/dashboard/pricing"
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-10 items-center justify-center rounded-xl border border-rent-border bg-rent-elevated px-4 text-xs font-semibold text-white/50 transition hover:border-rent-border-light hover:text-white/80 sm:h-12 sm:px-5 sm:text-sm"
        >
          Recharge Joules
        </a>
      </div>
    </div>
  );
}

export default BountyFallbackCard;
