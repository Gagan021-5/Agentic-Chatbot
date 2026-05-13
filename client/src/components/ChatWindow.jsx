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

function ChatWindow({ messages, isLoading, sendMessage, resetSession, sessionId = "" }) {
  const [input, setInput] = useState("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const viewportRef = useRef(null);

  useEffect(() => {
    if (!viewportRef.current) return;
    viewportRef.current.scrollTo({ top: viewportRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  // Cleanup speech recognition on unmount
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
        recognitionRef.current = null;
      }
    };
  }, []);

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

  /* ─── Web Speech API ─── */
  function startListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Your browser does not support voice input. Please use Chrome.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = "en-IN"; // Handles English, Hindi, and Hinglish mixed input
    recognitionRef.current = recognition;

    recognition.onstart = () => setIsListening(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput((prev) => (prev ? prev + " " + transcript : transcript));
    };

    recognition.onerror = (event) => {
      console.error("Speech recognition error:", event.error);
      setIsListening(false);

      if (event.error === "network") {
        alert(
          "Microphone network disconnected. Chrome requires an active internet connection to transcribe speech. If testing locally, this may be a Chrome HTTP restriction."
        );
      } else if (event.error === "not-allowed") {
        alert("Microphone access denied. Please allow permissions in your browser address bar.");
      }
    };

    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;
    };

    recognition.start();
  }

  function stopListening() {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  }

  function toggleListening() {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
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
            <MessageBubble
              key={msg.id}
              message={msg}
              sessionId={sessionId}
              onSendMessage={sendMessage}
              onResetSession={resetSession}
              isLoading={isLoading}
            />
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
          <input disabled={isLoading} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={handleKeyDown} placeholder="Describe your project, or attach a file." className="h-10 min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-white/30 sm:h-11 sm:text-[15px]" />

          {/* Mic Button — pulses red when listening */}
          <button
            type="button"
            onClick={toggleListening}
            disabled={isLoading}
            className={`inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition sm:h-10 sm:w-10 ${
              isListening
                ? "mic-pulse bg-red-500/20 text-red-400 shadow-lg shadow-red-500/20"
                : "text-white/40 hover:bg-white/5 hover:text-white/70"
            }`}
            title={isListening ? "Stop listening" : "Start voice input"}
          >
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
