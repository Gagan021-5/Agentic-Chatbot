/** In dev, use same-origin `/api` so Vite proxies to the server (see vite.config.js). */
const API_ORIGIN =
  typeof import.meta !== "undefined" && import.meta.env?.DEV
    ? ""
    : (import.meta.env?.VITE_API_ORIGIN ?? "http://localhost:3001");

function apiUrl(path) {
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${API_ORIGIN}${p}`;
}

export async function postAgentMessage(payload) {
  const response = await fetch(apiUrl("/api/agent/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || "Request failed");
  }

  return response.json();
}

export async function fetchAgentHistory(sessionId) {
  const response = await fetch(
    apiUrl(`/api/agent/history?sessionId=${encodeURIComponent(sessionId)}`)
  );
  
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || "Request failed");
  }

  return response.json();
}

export async function testPromptRun(payload) {
  const response = await fetch(apiUrl("/api/test-prompt"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || "Test run failed");
  }

  return response.json();
}

export async function testPreview(payload) {
  const response = await fetch(apiUrl("/api/test-preview"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || "Preview failed");
  }

  return response.json();
}
