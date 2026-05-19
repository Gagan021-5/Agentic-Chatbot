import { useEffect, startTransition, useState } from "react";
import { postAgentMessage, fetchAgentHistory } from "../utils/api";
import { formatUserPayloadForHistory, normalizeAgentUiData } from "../utils/formatUserHistoryDisplay";

const STORAGE_KEY = "rentprompts-agent-session-id";

function createSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function getInitialSessionId() {
  const existing = window.localStorage.getItem(STORAGE_KEY);

  if (existing) {
    return existing;
  }

  const nextSessionId = createSessionId();
  window.localStorage.setItem(STORAGE_KEY, nextSessionId);
  return nextSessionId;
}

function buildWelcomeMessage() {
  return {
    id: "welcome-message",
    role: "agent",
    text: "Hey! I'm the RentPrompts App Creation Agent.\n\nDescribe the AI app you want to build, even if it's rough. I'll turn it into a model recommendation, prompt template, SEO profile, and publish-ready demo flow.",
    uiType: "text",
    uiData: {}
  };
}

function useChat() {
  const [messages, setMessages] = useState([buildWelcomeMessage()]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(getInitialSessionId);

  useEffect(() => {
    async function loadHistory() {
      try {
        const data = await fetchAgentHistory(sessionId);
        if (data.history && data.history.length > 0) {
          const loadedMessages = data.history.map((msg, index) => {
            const raw = msg.content ?? msg.text ?? "";
            const isUser = msg.role === "user";

            if (isUser) {
              const friendly =
                (msg.displayContent && String(msg.displayContent).trim()) ||
                formatUserPayloadForHistory(raw) ||
                raw;
              return {
                id: `history-${index}`,
                role: "user",
                text: friendly,
                uiType: "text",
                uiData: {}
              };
            }

            const uiData = normalizeAgentUiData(msg.uiData);
            return {
              id: `history-${index}`,
              role: "agent",
              text: raw,
              uiType: msg.uiType || null,
              uiData,
              confirm: msg.confirm ?? null,
              step: msg.step,
              coins: msg.coins
            };
          });
          
          startTransition(() => {
            // Prepend welcome only once — never inject it mid-conversation
            setMessages((current) => {
              const hasOnlyWelcome = current.length === 1 && current[0].id === 'welcome-message';
              return hasOnlyWelcome
                ? [buildWelcomeMessage(), ...loadedMessages]
                : [buildWelcomeMessage(), ...loadedMessages]; // always rebuild cleanly from server truth
            });
          });
        }
      } catch (err) {
        console.error("Failed to load history:", err);
      }
    }
    loadHistory();
  }, [sessionId]);

  async function sendMessage(text) {
    const cleanText = String(text || "").trim();

    if (!cleanText || isLoading) {
      return;
    }

    let displayText = cleanText;
    const lowerText = cleanText.toLowerCase();
    
    // Hide structured payloads from the chat — show friendly text instead
    if (lowerText.startsWith("multi_select_form::")) {
      displayText = formatUserPayloadForHistory(cleanText) || "Confirmed settings ✓";
    } else if (lowerText.startsWith("confirm seo::")) {
      displayText = formatUserPayloadForHistory(cleanText) || "Confirmed SEO Metadata ✓";
    } else if (lowerText.startsWith("edit prompt::")) {
      displayText = formatUserPayloadForHistory(cleanText) || "Edited prompt template";
    } else if (lowerText.startsWith("select ")) {
      displayText = "Selected model ✓";
    } else if (lowerText === "publish app") {
      displayText = "Publishing app...";
    }

    const userMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      text: displayText
    };

    startTransition(() => {
      setMessages((current) => [...current, userMessage]);
    });

    setIsLoading(true);

    try {
      const data = await postAgentMessage({
        sessionId,
        message: cleanText
      });

      const agentMessage = {
        id: `${Date.now()}-agent`,
        role: "agent",
        text: data.reply,
        uiType: data.uiType,
        uiData: data.uiData,
        step: data.step,
        coins: data.coins,
        confirm: data.confirm || null
      };

      startTransition(() => {
        setMessages((current) => [...current, agentMessage]);
      });
    } catch (error) {
      startTransition(() => {
        setMessages((current) => [
          ...current,
          {
            id: `${Date.now()}-error`,
            role: "agent",
            text: "Something went wrong while talking to the agent. Check the server env values or try again.",
            uiType: "text",
            uiData: {}
          }
        ]);
      });
    } finally {
      setIsLoading(false);
    }
  }

  function resetSession() {
    const nextSessionId = createSessionId();
    window.localStorage.setItem(STORAGE_KEY, nextSessionId);
    setSessionId(nextSessionId);
    setIsLoading(false);
    startTransition(() => {
      setMessages([buildWelcomeMessage()]);
    });
  }

  return {
    messages,
    isLoading,
    sessionId,
    sendMessage,
    resetSession
  };
}

export default useChat;
