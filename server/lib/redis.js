import { Redis } from "@upstash/redis";

const SESSION_TTL_SECONDS = 1800;
const memoryStore = new Map();
const upstashUrl = process.env.UPSTASH_REDIS_REST_URL;
const upstashToken = process.env.UPSTASH_REDIS_REST_TOKEN;

function hasRealValue(value) {
  return Boolean(value && !/^your_.+_here$/i.test(value));
}

const redis =
  hasRealValue(upstashUrl) && hasRealValue(upstashToken) && /^https:\/\//i.test(upstashUrl)
    ? new Redis({
        url: upstashUrl,
        token: upstashToken
      })
    : null;

function getKey(sessionId) {
  return `session:${sessionId}`;
}

function createSession(sessionId) {
  return {
    sessionId,
    step: 0,
    awaitingConfirmation: false,
    confirmStep: null,
    awaitingDeepAnswer: false,
    currentDeepField: null,
    deepAnswers: {},
    appType: null,
    modelId: null,
    modelCost: null,
    extraction: null,
    promptData: null,
    scopeData: null,
    seoData: null,
    budgetPath: null,
    history: [],
    // Requirement gathering state machine
    requirements: {},
    currentField: null,
    userType: "unknown",
    enterpriseSignals: false
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function setMemory(sessionId, session) {
  memoryStore.set(getKey(sessionId), {
    expiresAt: Date.now() + SESSION_TTL_SECONDS * 1000,
    value: clone(session)
  });
}

function getMemory(sessionId) {
  const entry = memoryStore.get(getKey(sessionId));

  if (!entry) {
    return null;
  }

  if (entry.expiresAt < Date.now()) {
    memoryStore.delete(getKey(sessionId));
    return null;
  }

  return clone(entry.value);
}

async function getSession(sessionId) {
  if (!redis) {
    return getMemory(sessionId);
  }

  const session = await redis.get(getKey(sessionId));
  return session || null;
}

async function saveSession(session) {
  if (!redis) {
    setMemory(session.sessionId, session);
    return;
  }

  await redis.set(getKey(session.sessionId), session, {
    ex: SESSION_TTL_SECONDS
  });
}

async function deleteSession(sessionId) {
  if (!redis) {
    memoryStore.delete(getKey(sessionId));
    return;
  }

  await redis.del(getKey(sessionId));
}

export {
  SESSION_TTL_SECONDS,
  createSession,
  getSession,
  saveSession,
  deleteSession
};
