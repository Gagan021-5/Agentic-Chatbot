function OptionChips({ options, onSendMessage, isLoading }) {
  return (
    <div className="flex flex-wrap gap-2 sm:gap-2.5">
      {options.map((option) => (
        <button
          key={option}
          type="button"
          onClick={() => onSendMessage(option)}
          disabled={isLoading}
          className="rounded-full border border-rent-border bg-rent-elevated px-3.5 py-2 text-xs font-semibold text-white/85 transition-all hover:border-rent-purple/30 hover:bg-rent-purple/10 hover:text-white sm:px-4 sm:py-2.5 sm:text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {option}
        </button>
      ))}
    </div>
  );
}

export default OptionChips;
