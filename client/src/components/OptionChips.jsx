import React from "react";

function OptionChips({ options, onSendMessage, isLoading }) {
  // CRITICAL SAFETY GUARD: Prevents app crash if options are missing
  if (!options || !Array.isArray(options) || options.length === 0) {
    return null;
  }

  return (
    // Global Options Container: Vertical Stack
    <div className="flex flex-col gap-2.5 mt-3 w-full max-w-md animate-fade-in-up">
      {options.map((option, i) => (
        <button
          key={i}
          onClick={() => onSendMessage(option)}
          disabled={isLoading}
          className={`
            text-left w-full px-5 py-3.5 
            bg-gradient-to-br from-[#050505] to-[#121018]
            border border-white/[0.06] rounded-2xl 
            text-sm font-medium text-gray-200
            
            transition-all duration-300 ease-in-out
            
            hover:border-purple-500/40 
            hover:shadow-[0_0_20px_rgba(168,85,247,0.2)]
            hover:bg-[#0f0b17]
            hover:-translate-y-1
            
            active:scale-95
            
            disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:translate-y-0 disabled:hover:shadow-none
            
            flex justify-between items-center group
          `}
        >
          <span className="tracking-[0.01em]">{option}</span>

          <span
            className="
            opacity-0 -translate-x-2
            group-hover:opacity-100 group-hover:translate-x-0
            transition-all duration-300 ease-out 
            text-[#8b5cf6] font-bold
          "
          >
            →
          </span>
        </button>
      ))}
    </div>
  );
}

export default OptionChips;
