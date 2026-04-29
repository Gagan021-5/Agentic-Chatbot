import { useEffect, useMemo, useRef, useState } from "react";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

function IconArrowLeft() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
      <path d="M12.5 4.5L7 10l5.5 5.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconPlus() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
      <path d="M10 4.5v11M4.5 10h11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function IconRefresh() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
      <path d="M15.5 7.3A6.5 6.5 0 1 0 16 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M12.5 4.6h3v3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconPaperclip() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
      <path d="M7.2 10.8l4.7-4.7a2.5 2.5 0 1 1 3.5 3.5l-6 6a4 4 0 1 1-5.7-5.7l6.2-6.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconMic() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
      <rect x="7" y="3.5" width="6" height="9" rx="3" stroke="currentColor" strokeWidth="1.6" />
      <path d="M5.5 9.5a4.5 4.5 0 0 0 9 0M10 14v2.5M7.5 16.5h5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5">
      <path d="M4 10l12-6-4 12-3-5-5-1z" fill="currentColor" />
    </svg>
  );
}




function ChatWindow({ messages, isLoading, sendMessage, resetSession }) {
  const [input, setInput] = useState("");
  const viewportRef = useRef(null);

  useEffect(() => {
    if (!viewportRef.current) return;
    viewportRef.current.scrollTo({ top: viewportRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  const memoryLevel = useMemo(() => Math.min(91, 63 + messages.length * 2), [messages.length]);

  function handleSubmit(e) {
    e.preventDefault();
    if (!input.trim()) return;
    sendMessage(input);
    setInput("");
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) handleSubmit(e);
  }

  return (
    <div className="relative flex h-screen flex-col overflow-hidden bg-rent-bg" style={{ height: "100dvh" }}>
      {/* Header */}
      <header className="glass-panel relative z-30 shrink-0 border-b border-rent-border/60">
        <div className="mx-auto flex h-14 max-w-[1400px] items-center justify-between gap-2 px-3 sm:h-16 sm:gap-3 sm:px-5 md:px-8">
          <button className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm font-medium text-white/60 transition hover:bg-white/5 hover:text-white sm:gap-2 sm:px-3" type="button">
            <IconArrowLeft />
            <span className="hidden sm:inline">Back</span>
          </button>

          <div className="flex items-center gap-2 sm:gap-3">
            <span className="relative flex h-2 w-2 sm:h-2.5 sm:w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rent-green opacity-40" />
              <span className="relative inline-flex h-full w-full rounded-full bg-rent-green" />
            </span>
            <span className="text-sm font-bold text-white sm:text-base">RentPrompts Agent</span>
            <span className="hidden rounded-full border border-rent-border bg-rent-surface px-2.5 py-0.5 text-[11px] font-semibold text-white/50 sm:inline-flex sm:px-3 sm:py-1 sm:text-xs">
              Agent Memory: {memoryLevel}%
            </span>
          </div>

          <div className="flex items-center gap-2 sm:gap-2.5">
            <button type="button" onClick={resetSession} className="inline-flex h-9 items-center gap-1.5 rounded-xl border border-rent-border bg-rent-surface px-2.5 text-xs font-semibold text-white transition hover:border-rent-border-light hover:bg-rent-elevated sm:h-10 sm:gap-2 sm:px-3.5 sm:text-sm">
              <IconPlus />
              <span className="hidden sm:inline">New Chat</span>
            </button>
            <button type="button" onClick={resetSession} className="hidden h-10 w-10 items-center justify-center rounded-xl border border-rent-border bg-rent-surface text-white/50 transition hover:text-white md:inline-flex">
              <IconRefresh />
            </button>
            <button type="button" disabled className="hidden h-10 items-center justify-center rounded-xl border border-rent-border bg-rent-surface px-5 text-sm font-semibold text-white/30 lg:inline-flex">Save Draft</button>
            <button type="button" disabled className="hidden h-10 items-center justify-center rounded-xl bg-rent-purple px-5 text-sm font-bold text-white lg:inline-flex">Publish</button>
          </div>
        </div>
      </header>

      {/* Messages */}
      <main ref={viewportRef} className="relative z-10 flex-1 overflow-y-auto px-3 py-5 sm:px-5 sm:py-8 md:px-8">
        <div className="mx-auto flex max-w-[960px] flex-col gap-5 sm:gap-7 md:gap-8">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} onSendMessage={sendMessage} onResetSession={resetSession} />
          ))}
          {isLoading ? <TypingIndicator /> : null}
        </div>
      </main>

      {/* Input Bar */}
      <div className="relative z-30 shrink-0 border-t border-rent-border/40 bg-rent-bg/80 px-3 pb-3 pt-3 backdrop-blur-xl sm:px-5 sm:pb-5 sm:pt-4 md:px-8 safe-bottom">
        <form onSubmit={handleSubmit} className="input-focus-ring mx-auto flex max-w-[720px] items-center gap-2 rounded-2xl border border-rent-border bg-rent-surface/90 px-3 py-2.5 shadow-soft transition-all sm:gap-3 sm:rounded-[22px] sm:px-4 sm:py-3 lg:max-w-[820px]">
          <button type="button" className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-white/40 transition hover:bg-white/5 hover:text-white/70 sm:h-10 sm:w-10">
            <IconPaperclip />
          </button>
          <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="Describe your project, or attach a file." className="h-10 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30 sm:h-11 sm:text-[15px]" />
          <button type="button" className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-xl text-white/40 transition hover:bg-white/5 hover:text-white/70 sm:inline-flex sm:h-10 sm:w-10">
            <IconMic />
          </button>
          <button type="submit" disabled={isLoading || !input.trim()} className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-rent-purple text-white shadow-lg shadow-rent-purple/20 transition hover:bg-[#8b6ffe] hover:shadow-rent-purple/30 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none sm:h-10 sm:w-10">
            <IconSend />
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatWindow;
