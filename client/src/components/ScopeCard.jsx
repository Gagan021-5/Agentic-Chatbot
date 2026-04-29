function ScopeCard({ data, onSendMessage }) {
  if (!data || !data.items) return null;

  function getComplexityStyle(complexity) {
    if (complexity === 'simple') return 'bg-[#14532d] text-[#4ade80]';
    if (complexity === 'medium') return 'bg-[#1e3a5f] text-[#60a5fa]';
    if (complexity === 'complex') return 'bg-[#450a0a] text-[#f87171]';
    return 'bg-[#2a2a2a] text-[#888]';
  }

  function getPriorityStyle(priority) {
    if (priority === 'Must Have') return 'bg-[#422006] text-[#fb923c]';
    if (priority === 'Should Have') return 'bg-[#1e1e2e] text-[#818cf8]';
    if (priority === 'Nice to Have') return 'bg-[#1a1a1a] text-[#888] border border-[#333]';
    return 'bg-[#2a2a2a] text-[#888]';
  }

  return (
    <div className="mt-2">
      <div className="rounded-xl border border-[#2a2a2a] bg-[#1a1a1a] p-4">
        {/* Header */}
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm font-bold text-white">Scope Summary</div>
          <div className="flex gap-2">
            <span className="rounded-full bg-[#2a2a2a] px-2.5 py-1 text-[10px] font-semibold text-[#ccc]">
              {data.totalItems} items
            </span>
            <span className="rounded-full bg-[#2a2a2a] px-2.5 py-1 text-[10px] font-semibold text-[#ccc]">
              ~{data.totalHours}h
            </span>
          </div>
        </div>

        {/* Scope Items */}
        <div className="flex flex-col">
          {data.items.map((item, index) => (
            <div
              key={index}
              className={`flex flex-col gap-3 py-4 sm:flex-row sm:items-start sm:justify-between ${
                index !== 0 ? 'border-t border-[#2a2a2a]' : 'border-t border-[#2a2a2a]'
              }`}
            >
              {/* Left Side */}
              <div className="min-w-0 flex-1 pr-4">
                <div className="text-sm font-medium text-white">{item.title}</div>
                <div className="mt-1 line-clamp-2 text-xs leading-5 text-[#666]">
                  {item.description}
                </div>
              </div>

              {/* Right Side (Badges) */}
              <div className="flex shrink-0 flex-wrap items-center gap-2 sm:justify-end">
                {item.aiAssisted && (
                  <span className="rounded bg-[#2e1065] px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-[#c084fc]">
                    AI ✦
                  </span>
                )}
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${getComplexityStyle(
                    item.complexity
                  )}`}
                >
                  {item.complexity}
                </span>
                <span
                  className={`rounded px-1.5 py-0.5 text-[10px] font-medium tracking-wide ${getPriorityStyle(
                    item.priority
                  )}`}
                >
                  {item.priority}
                </span>
                <span className="ml-1 text-xs font-semibold text-[#888]">
                  ~{item.estimatedHours}h
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="mt-2 border-t border-[#2a2a2a] pt-3 text-right text-xs text-[#888]">
          Total: ~{data.totalHours}h estimated
        </div>
      </div>

      {/* Action Chips Below Card */}
      <div className="mt-3 flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => onSendMessage("Looks good")}
          className="rounded-full bg-gradient-to-r from-orange-500 to-pink-500 px-4 py-2 text-xs font-bold text-white transition hover:opacity-90"
        >
          Looks good, show pricing →
        </button>
        <button
          type="button"
          onClick={() => onSendMessage("I want to adjust something")}
          className="rounded-full bg-[#2a2a2a] px-4 py-2 text-xs font-semibold text-[#ccc] transition hover:bg-[#333] hover:text-white"
        >
          I want to adjust something
        </button>
      </div>
    </div>
  );
}

export default ScopeCard;
