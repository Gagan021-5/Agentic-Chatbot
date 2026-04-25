function PublishSuccessCard({ data, onResetSession }) {
  return (
    <div className="rounded-2xl border border-green-500/30 bg-gradient-to-b from-green-500/8 to-rent-bg/95 p-4 shadow-soft sm:rounded-[22px] sm:p-6">
      <div className="flex items-start gap-3 sm:gap-4">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl border border-green-500/25 bg-green-500/10 text-xl text-green-300 sm:h-14 sm:w-14 sm:text-2xl">
          ✓
        </div>

        <div className="min-w-0 flex-1">
          <h3 className="text-lg font-extrabold text-white sm:text-2xl">{data.appName}</h3>
          <p className="mt-1.5 text-xs leading-6 text-green-100/60 sm:mt-2 sm:text-sm sm:leading-7">
            The mock publish payload has been logged from the server and the session has been cleared.
          </p>

          <div className="mt-4 grid gap-2 sm:mt-5 sm:gap-3 md:grid-cols-3">
            <div className="rounded-xl border border-white/6 bg-black/15 p-3 sm:rounded-2xl sm:p-4">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/35 sm:text-xs">Model</div>
              <div className="mt-1.5 text-xs font-semibold text-white sm:mt-2 sm:text-sm">{data.modelId}</div>
            </div>
            <div className="rounded-xl border border-white/6 bg-black/15 p-3 sm:rounded-2xl sm:p-4">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/35 sm:text-xs">Cost per run</div>
              <div className="mt-1.5 text-xs font-semibold text-white sm:mt-2 sm:text-sm">{data.costPerRun} coins</div>
            </div>
            <div className="rounded-xl border border-white/6 bg-black/15 p-3 sm:rounded-2xl sm:p-4">
              <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-white/35 sm:text-xs">Plan</div>
              <div className="mt-1.5 text-xs font-semibold capitalize text-white sm:mt-2 sm:text-sm">{data.selectedPlan}</div>
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-1.5 sm:mt-4 sm:gap-2">
            {(data.tags || []).map((tag) => (
              <span key={tag} className="rounded-full border border-green-500/15 bg-green-500/8 px-2.5 py-1 text-[10px] font-semibold text-green-100/70 sm:px-3 sm:py-1.5 sm:text-xs">{tag}</span>
            ))}
          </div>

          <div className="mt-4 flex flex-col gap-2 sm:mt-5 sm:flex-row sm:gap-3">
            <a href={data.mockUrl} target="_blank" rel="noreferrer" className="inline-flex h-10 items-center justify-center rounded-xl border border-green-500/15 bg-green-500/8 px-4 text-xs font-bold text-green-100 transition hover:bg-green-500/12 sm:h-12 sm:px-5 sm:text-sm">
              Open mock marketplace link
            </a>
            <button type="button" onClick={onResetSession} className="btn-cta h-10 rounded-xl px-4 text-xs font-bold text-white sm:h-12 sm:px-5 sm:text-sm">
              Create Another App
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default PublishSuccessCard;
