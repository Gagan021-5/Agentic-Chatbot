import BudgetCards from "./BudgetCards";
import ScopeCard from "./ScopeCard";
import BountyFallbackCard from "./BountyFallbackCard";
import ConfirmCard from "./ConfirmCard";
import CostWarningCard from "./CostWarningCard";
import InlineTextInput from "./InlineTextInput";
import ModelCard from "./ModelCard";
import OptionChips from "./OptionChips";
import PromptPreviewCard from "./PromptPreviewCard";
import PublishSuccessCard from "./PublishSuccessCard";
import RequirementSummaryCard from "./RequirementSummaryCard";
import SEOPreviewCard from "./SEOPreviewCard";
import AppPreviewCard from "./AppPreviewCard";
import MultiSelectFormCard from "./MultiSelectFormCard";
function IconCopy() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5">
      <rect x="6" y="6" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1.5" />
      <path d="M14 6V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h1" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function IconEdit() {
  return (
    <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5">
      <path d="m4.5 14.8 1.1-3.4L12.7 4.3a1.4 1.4 0 0 1 2 0l1 1a1.4 1.4 0 0 1 0 2l-7.1 7.1-3.4 1.1Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function Avatar({ role }) {
  const isUser = role === "user";
  return (
    <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-xs font-bold ring-1 sm:h-10 sm:w-10 ${
      isUser
        ? "bg-rent-elevated text-white/90 ring-rent-border-light"
        : "bg-rent-purple/15 text-rent-purple ring-rent-purple/20"
    }`}>
      {isUser ? "G" : "RP"}
    </div>
  );
}

function renderText(text) {
  return text.split("\n").map((line, i) => {
    const parts = line.split(/(\*\*.*?\*\*)/g);

    return (
      <p key={`${i}-${line.slice(0, 12)}`} className={i === 0 ? "" : "mt-3"}>
        {parts.map((part, j) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            return (
              <strong key={j} className="font-bold text-white">
                {part.slice(2, -2)}
              </strong>
            );
          }
          return part;
        })}
      </p>
    );
  });
}

function AgentUI(props) {
  const { message, onSendMessage, onResetSession, isLoading } = props;
  const previewSessionId = props.sessionId ?? "";
  const { uiType, uiData } = message;
  if (uiType === "chips") return <OptionChips options={uiData.options || []} onSendMessage={onSendMessage} isLoading={isLoading} />;
  if (uiType === "text_input") return <InlineTextInput uiData={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "models") return (
    <div className="grid gap-3 sm:gap-4 lg:grid-cols-3">
      {(uiData.models || []).map((m) => <ModelCard key={m.id} model={m} onSendMessage={onSendMessage} isLoading={isLoading} />)}
    </div>
  );
  if (uiType === "cost_warning") return <CostWarningCard data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "prompt_preview") return <PromptPreviewCard data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "seo_preview") return <SEOPreviewCard data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "budget_cards") return <BudgetCards data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "bounty_fallback") return <BountyFallbackCard data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "success") return <PublishSuccessCard data={uiData} onResetSession={onResetSession} />;
  if (uiType === "scope") return <ScopeCard data={uiData} onSendMessage={onSendMessage} />;
  if (uiType === "confirm") return <RequirementSummaryCard data={uiData} />;
  if (uiType === "app_preview")
    return (
      <AppPreviewCard
        data={uiData}
        onSendMessage={onSendMessage}
        sessionId={previewSessionId}
        storageMessageId={message.id}
      />
    );
  if (uiType === "multi_select_form") return <MultiSelectFormCard data={uiData} onSendMessage={onSendMessage} isLoading={isLoading} />;
  return null;
}

function MessageBubble({ message, onSendMessage, onResetSession, isLoading, sessionId = "" }) {
  const isUser = message.role === "user";

  // Clean up the displayed text so the user's bubble looks natural
  let displayText = message.text || "";
  if (isUser && displayText.toLowerCase().startsWith("select ")) {
    displayText = `I choose ${displayText.slice(7).trim()}`;
  }

  return (
    <div className={`message-enter flex items-start gap-2.5 sm:gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar role={message.role} />

      <div className={`min-w-0 max-w-[calc(100%-52px)] sm:max-w-[calc(100%-60px)] ${isUser ? "" : "flex-1"}`}>
        <div className={`rounded-2xl border px-4 py-3 shadow-soft sm:rounded-[22px] sm:px-5 sm:py-4 ${
          isUser
            ? "border-rent-border-light bg-rent-elevated text-white"
            : "glass-panel border-rent-border bg-rent-card/80 text-white/90"
        }`}>
          {displayText ? (
            <div className="text-[13px] leading-7 sm:text-[14.5px] sm:leading-8">
              {renderText(displayText)}
            </div>
          ) : null}
          {!isUser && message.uiType && message.uiType !== "text" ? (
            <div className={displayText ? "mt-4 sm:mt-5" : ""}>
              <AgentUI
                message={message}
                onSendMessage={onSendMessage}
                onResetSession={onResetSession}
                isLoading={isLoading}
                sessionId={sessionId}
              />
            </div>
          ) : null}
          {/* ConfirmCard — shown after every agent step that needs confirmation */}
          {!isUser && message.confirm ? (
            <ConfirmCard
              summary={message.confirm.summary}
              detail={message.confirm.detail}
              onYes={() => onSendMessage("Yes, proceed")}
              onNo={(correction) => onSendMessage(`Change: ${correction}`)}
            />
          ) : null}
        </div>

        {/* Action buttons */}
        <div className={`mt-1.5 flex items-center gap-1 ${isUser ? "justify-end" : ""}`}>
          {!isUser ? (
            <button type="button" className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/50">
              <IconCopy />
            </button>
          ) : null}
          {isUser ? (
            <button type="button" className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/50">
              <IconEdit />
            </button>
          ) : null}
          {isUser ? (
            <button type="button" className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/50">
              <IconCopy />
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;
