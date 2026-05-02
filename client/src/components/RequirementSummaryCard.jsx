function RequirementSummaryCard({ data }) {
  if (!data) return null;

  return (
    <div className="space-y-3">
      {/* App type badge */}
      {data.summary ? (
        <div className="inline-flex items-center gap-2 rounded-full border border-rent-purple/25 bg-rent-purple/10 px-3.5 py-1.5">
          <span className="text-xs font-bold tracking-wide text-rent-purple">
            {data.summary}
          </span>
        </div>
      ) : null}

      {/* Requirement fields */}
      {data.detail ? (
        <div className="rounded-xl border border-rent-border/60 bg-rent-surface/60 p-3.5 sm:p-4">
          {data.detail.split('\n').map((line, i) => (
            <p
              key={`req-${i}`}
              className={`text-[13px] leading-6 text-white/70 sm:text-sm sm:leading-7 ${
                i > 0 ? 'mt-1' : ''
              }`}
            >
              {line}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default RequirementSummaryCard;
