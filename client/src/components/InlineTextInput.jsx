import { useState } from "react";

function InlineTextInput({ uiData = {}, onSendMessage }) {
  const [value, setValue] = useState("");
  const placeholder = uiData?.placeholder || "Type your answer...";

  function handleSubmit(e) {
    e.preventDefault();
    const cleanValue = value.trim();
    if (!cleanValue) return;
    onSendMessage(cleanValue);
    setValue("");
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      {uiData?.progress ? (
        <p className="text-[11px] font-medium text-[#666]">
          {uiData.progress}
        </p>
      ) : null}

      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          className="min-h-11 min-w-0 flex-1 rounded-lg border border-[#333] bg-[#2a2a2a] px-3.5 py-2.5 text-sm text-white outline-none placeholder:text-white/25 transition focus:border-orange-400/45 focus:ring-2 focus:ring-orange-400/10"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="min-h-11 rounded-lg bg-gradient-to-r from-orange-500 to-pink-500 px-4 text-sm font-bold text-white transition hover:from-orange-400 hover:to-pink-400 disabled:cursor-not-allowed disabled:opacity-45 sm:px-5"
        >
          Send
        </button>
      </div>
    </form>
  );
}

export default InlineTextInput;
