export async function postAgentMessage(payload) {
  const response = await fetch("http://localhost:3001/api/agent/chat", {
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
  const response = await fetch(`http://localhost:3001/api/agent/history?sessionId=${encodeURIComponent(sessionId)}`);
  
  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || "Request failed");
  }

  return response.json();
}

export async function testPromptRun(payload) {
  const response = await fetch("http://localhost:3001/api/test-prompt", {
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
