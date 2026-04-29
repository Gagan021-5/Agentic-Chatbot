function PlanCard({ planKey, plan, featured, onSendMessage }) {
  return (
    <div className={`card-hover relative flex h-full flex-col rounded-2xl border p-4 shadow-soft sm:rounded-[22px] sm:p-5 ${
      featured
        ? "border-rent-purple/40 bg-gradient-to-b from-rent-purple/12 to-rent-bg/95"
        : "border-rent-border bg-rent-card"
    }`}>
      {featured ? (
        <div className="absolute -top-2.5 left-4 rounded-full bg-rent-purple px-3 py-0.5 text-[10px] font-bold uppercase tracking-[0.18em] text-white sm:left-5 sm:px-4 sm:py-1 sm:text-xs">
          Recommended
        </div>
      ) : null}

      <div className="mt-2 flex flex-col gap-4">
        {/* Header & Price - wraps automatically on small screens */}
        <div className="flex flex-wrap items-start justify-between gap-3">
          <h3 className="text-xl font-extrabold text-white sm:text-2xl">{plan.label}</h3>
          
          <div className="shrink-0 rounded-2xl border border-white/8 bg-black/20 px-3 py-2 text-right">
            <div className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-yellow-500 sm:h-2.5 sm:w-2.5" />
              <span className="text-xl font-extrabold text-white sm:text-2xl">{plan.joules.toLocaleString()}</span>
            </div>
            <div className="mt-1 flex items-center justify-end gap-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-white/40">
              <span>joules</span>
              <span className="normal-case tracking-normal text-white/45">(${plan.usd})</span>
            </div>
          </div>
        </div>

        <p className="text-sm leading-6 text-white/50 sm:text-sm sm:leading-7">{plan.desc}</p>
      </div>

      <div className="mt-4 flex flex-col gap-2 sm:mt-5">
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-1 text-[11px] text-white/60 sm:px-3 sm:text-xs">⊙ 7 scope items</span>
          <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-1 text-[11px] text-white/60 sm:px-3 sm:text-xs">⊙ 1 week</span>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full border border-white/8 bg-white/4 px-2.5 py-1 text-[11px] text-white/60 sm:px-3 sm:text-xs">⊙ AI-assisted with human review</span>
        </div>
      </div>

      <div className="mt-auto pt-4 sm:pt-5">
        <div className="rounded-xl border border-white/6 bg-black/15 p-3 sm:rounded-2xl sm:p-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-white/35 sm:text-xs">Why it fits</div>
          <p className="mt-2 text-xs leading-5 text-white/60 sm:text-[13px] sm:leading-6">
            {featured
              ? "This is the best balance of delivery quality, scope coverage, and current market positioning."
              : planKey === "lean"
                ? "Keeps must-have work and highest-priority scope items."
                : "Keeps the full requested scope intact."}
          </p>
        </div>

        <button
          type="button"
          onClick={() => onSendMessage(`Select ${planKey}`)}
          className={`mt-3 w-full rounded-xl py-2.5 text-sm font-bold transition sm:mt-4 sm:py-3 ${
            featured
              ? "btn-cta text-white"
              : "border border-rent-border bg-rent-elevated text-white hover:border-rent-border-light hover:bg-rent-elevated"
          }`}
        >
          Select Plan
        </button>
      </div>
    </div>
  );
}

function BudgetCards({ data, onSendMessage }) {
  const { options, context } = data;

  return (
    <div className="space-y-4 sm:space-y-5">
      <div className="grid gap-3 sm:gap-4 xl:grid-cols-[1.5fr_1fr]">
        <div className="rounded-2xl border border-rent-purple/25 bg-gradient-to-b from-rent-purple/15 to-rent-bg/95 p-4 shadow-soft sm:rounded-[22px] sm:p-6">
          <div className="text-[10px] font-semibold uppercase tracking-[0.25em] text-white/40 sm:text-xs">Budget options</div>
          <p className="mt-3 text-lg font-medium leading-8 text-white/85 sm:mt-4 sm:text-xl sm:leading-9">
            Pick a plan, or set a target budget and let ARIA reshape the first option around it.
          </p>
          <div className="mt-4 flex flex-wrap gap-2 text-xs text-white/55 sm:mt-5 sm:gap-3 sm:text-sm">
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 sm:px-4 sm:py-2">Floor: {context.floorJoules.toLocaleString()} Joules</span>
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 sm:px-4 sm:py-2">Market: {context.marketRange}</span>
          </div>
        </div>

        <div className="rounded-2xl border border-rent-border bg-gradient-to-b from-white/[0.03] to-rent-bg/95 p-4 shadow-soft sm:rounded-[22px] sm:p-6">
          <div className="text-[10px] font-semibold uppercase tracking-[0.25em] text-white/40 sm:text-xs">Budget target</div>
          <div className="mt-4 flex items-center gap-2 sm:gap-3">
            <div className="flex flex-1 items-center gap-2 rounded-xl border border-rent-border bg-rent-surface px-3 py-2 sm:px-4 sm:py-2.5">
              <span className="h-3 w-3 rounded-full bg-yellow-500 sm:h-3.5 sm:w-3.5" />
              <span className="text-xs text-white/40 sm:text-sm">Enter amount</span>
              <span className="ml-auto text-[10px] font-semibold uppercase tracking-wider text-white/30 sm:text-xs">Joules</span>
            </div>
            <button type="button" className="btn-cta h-9 shrink-0 rounded-xl px-4 text-xs font-bold text-white sm:h-10 sm:px-5 sm:text-sm">Apply Budget</button>
          </div>
          <p className="mt-3 text-xs leading-5 text-white/40 sm:text-sm sm:leading-6">
            Locks in this amount and moves straight to budget review — no extra options shown.
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:gap-4 xl:grid-cols-3">
        <PlanCard planKey="lean" plan={options.lean} onSendMessage={onSendMessage} />
        <PlanCard planKey="recommended" plan={options.recommended} featured onSendMessage={onSendMessage} />
        <PlanCard planKey="full" plan={options.full} onSendMessage={onSendMessage} />
      </div>
    </div>
  );
}

export default BudgetCards;
