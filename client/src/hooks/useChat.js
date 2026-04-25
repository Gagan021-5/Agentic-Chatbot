import { startTransition, useState } from "react";
import { postAgentMessage } from "../utils/api";

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

  async function sendMessage(text) {
    const cleanText = String(text || "").trim();

    if (!cleanText || isLoading) {
      return;
    }

    const userMessage = {
      id: `${Date.now()}-user`,
      role: "user",
      text: cleanText
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
        coins: data.coins
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
