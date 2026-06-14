import { useState } from "react";
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
import FinalBlueprintCard from "./FinalBlueprintCard";
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

function renderInline(text) {
  // Split on **bold**, *italic*, and `code` patterns
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*|`[^`]+`)/g);
  return parts.map((part, j) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={j} className="font-bold text-white">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={j} className="italic text-white/80">{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={j} className="rounded bg-white/10 px-1.5 py-0.5 text-[12px] font-mono text-purple-300">{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function renderText(text) {
  const lines = text.split("\n");
  const elements = [];
  let listBuffer = [];
  let listType = null; // 'ul' or 'ol'

  function flushList() {
    if (listBuffer.length === 0) return;
    if (listType === 'ol') {
      elements.push(
        <ol key={`ol-${elements.length}`} className="mt-2 ml-4 list-decimal space-y-1 text-gray-300">
          {listBuffer.map((item, li) => <li key={li} className="pl-1">{renderInline(item)}</li>)}
        </ol>
      );
    } else {
      elements.push(
        <ul key={`ul-${elements.length}`} className="mt-2 ml-4 space-y-1 text-gray-300">
          {listBuffer.map((item, li) => (
            <li key={li} className="flex items-start gap-2">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-rent-purple/60" />
              <span>{renderInline(item)}</span>
            </li>
          ))}
        </ul>
      );
    }
    listBuffer = [];
    listType = null;
  }

  lines.forEach((line, i) => {
    const trimmed = line.trim();

    // Horizontal rule / divider
    if (/^-{3,}$/.test(trimmed) || /^_{3,}$/.test(trimmed)) {
      flushList();
      elements.push(<hr key={`hr-${i}`} className="my-3 border-white/10" />);
      return;
    }

    // Heading: ## or ###
    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(
        <h4 key={`h4-${i}`} className="mt-3 mb-1 text-[13px] font-bold uppercase tracking-wide text-white/50 sm:text-[13.5px]">
          {renderInline(trimmed.slice(4))}
        </h4>
      );
      return;
    }
    if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(
        <h3 key={`h3-${i}`} className="mt-3 mb-1 text-[14px] font-bold text-white sm:text-[15px]">
          {renderInline(trimmed.slice(3))}
        </h3>
      );
      return;
    }

    // Bullet list: - item, • item, * item (but not bold **)
    const bulletMatch = trimmed.match(/^[-•*]\s+(.+)/);
    if (bulletMatch && !trimmed.startsWith("**")) {
      if (listType !== 'ul') flushList();
      listType = 'ul';
      listBuffer.push(bulletMatch[1]);
      return;
    }

    // Numbered list: 1. item, 2) item
    const numMatch = trimmed.match(/^\d+[.)]\s+(.+)/);
    if (numMatch) {
      if (listType !== 'ol') flushList();
      listType = 'ol';
      listBuffer.push(numMatch[1]);
      return;
    }

    // Empty line
    if (!trimmed) {
      flushList();
      elements.push(<div key={`sp-${i}`} className="h-2" />);
      return;
    }

    // Regular paragraph
    flushList();
    elements.push(
      <p key={`p-${i}`} className={elements.length === 0 ? "" : "mt-2"}>
        {renderInline(trimmed)}
      </p>
    );
  });

  flushList();
  return elements;
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
  if (uiType === "final_blueprint") return <FinalBlueprintCard data={uiData} />;
  return null;
}

// Prefixes that are internal signals to the backend — never show as raw chat bubbles
const HIDDEN_PAYLOAD_PREFIXES = [
  "SEO_PUBLISH::",
  "SEO_DRAFT::",
  "SEO_EDIT::",
  "multi_select_form::",
  "edit prompt::",
  "confirm seo::",
];

function MessageBubble({ message, onSendMessage, onResetSession, onEditMessage, isLoading, sessionId = "" }) {
  const isUser = message.role === "user";
  const [copied, setCopied] = useState(false);

  // Hide structured payload messages entirely — they're backend signals, not user text
  const rawText = message.text || "";
  if (isUser && HIDDEN_PAYLOAD_PREFIXES.some(p => rawText.startsWith(p))) {
    return null;
  }

  // Clean up the displayed text so the user's bubble looks natural
  let displayText = rawText;
  if (isUser && displayText.toLowerCase().startsWith("select ")) {
    displayText = `I choose ${displayText.slice(7).trim()}`;
  }

  function handleCopy(textToCopy) {
    const plain = String(textToCopy || "").replace(/\*\*/g, "").replace(/\*/g, "").trim();
    if (!plain) return;
    navigator.clipboard.writeText(plain).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {
      // Fallback for browsers without clipboard API
      const ta = document.createElement("textarea");
      ta.value = plain;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  function handleEdit() {
    // Put the raw original text back into the input so the user can modify + resend
    const rawText = message.text || "";
    if (onEditMessage) onEditMessage(rawText);
  }

  return (
    <div className={`message-enter flex items-start gap-2.5 sm:gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <Avatar role={message.role} />

      <div className={`min-w-0 max-w-[calc(100%-52px)] sm:max-w-[calc(100%-60px)] ${isUser ? "" : "flex-1"}`}>
        <div className={`rounded-3xl border px-4 py-3.5 shadow-soft sm:px-5 sm:py-4 ${
          isUser
            ? "rounded-br-lg border-white/10 bg-rent-elevated text-gray-200"
            : "rounded-bl-lg glass-panel border-purple-500/15 bg-rent-card/80 text-gray-300"
        }`}>
          {displayText ? (
            <div className="text-[13px] leading-relaxed sm:text-[14.5px] sm:leading-loose">
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
          {/* Agent message: copy button */}
          {!isUser ? (
            <button
              type="button"
              title="Copy reply"
              onClick={() => handleCopy(message.text)}
              className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/60 active:scale-90"
            >
              {copied ? (
                <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5 text-green-400">
                  <path d="M4 10l4.5 4.5L16 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <IconCopy />
              )}
            </button>
          ) : null}

          {/* User message: edit button — pre-fills input with original text */}
          {isUser ? (
            <button
              type="button"
              title="Edit message"
              onClick={handleEdit}
              className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/60 active:scale-90"
            >
              <IconEdit />
            </button>
          ) : null}

          {/* User message: copy button */}
          {isUser ? (
            <button
              type="button"
              title="Copy message"
              onClick={() => handleCopy(message.text)}
              className="inline-flex h-7 w-7 items-center justify-center rounded-lg text-white/25 transition hover:bg-white/5 hover:text-white/60 active:scale-90"
            >
              {copied ? (
                <svg viewBox="0 0 20 20" fill="none" className="h-3.5 w-3.5 text-green-400">
                  <path d="M4 10l4.5 4.5L16 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <IconCopy />
              )}
            </button>
          ) : null}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;
