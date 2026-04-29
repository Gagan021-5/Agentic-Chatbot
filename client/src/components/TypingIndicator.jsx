
function TypingIndicator() {
  return (
    <div className="message-enter flex items-start gap-3 px-1 sm:px-2">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-rent-purple/15 text-xs font-bold text-rent-purple ring-1 ring-rent-purple/20 sm:h-10 sm:w-10">
        RP
      </div>
      <div className="glass-panel rounded-2xl border border-rent-border bg-rent-card/90 px-4 py-3 shadow-soft">
        <div className="flex items-center gap-3 text-sm text-white/70">
          <span>Agent is thinking</span>
          <div className="flex items-center gap-1.5">
            <span className="dot-pulse h-1.5 w-1.5 rounded-full bg-rent-purple" />
            <span className="dot-pulse h-1.5 w-1.5 rounded-full bg-rent-purple" />
            <span className="dot-pulse h-1.5 w-1.5 rounded-full bg-rent-purple" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default TypingIndicator;
